# exp15_pp_schedules.py —— 实验二：三种流水线调度对拍——naive / GPipe / 1F1B
# 同一个 12 层 MLP 切 2 段，同一份初始权重、同一份数据，三种调度各跑 5 步：
#   naive：整个 batch 一次流过（m=1）——任何时刻只有一段在算
#   GPipe：batch 切成 m=4 份，全部前向完再全部反向
#   1F1B ：预热后前向反向交替——气泡与 GPipe 相同，在途激活从 m 份降到 p 份
# 看三组数字：每 step 耗时、recv 空等时间（气泡的实测）、在途激活峰值（份）。
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"
torch.set_num_threads(1)    # 每进程限一份算力，流水线重叠的收益才能体现
                            #（GPU 上天然如此：一张卡就是一份算力）

# ---- 通信计数器：按 op 记次数与本地缓冲字节数，打印后清零 ----
_CNT = {}
def _rec(op, t):
    _CNT.setdefault(op, [0, 0])
    _CNT[op][0] += 1
    _CNT[op][1] += t.numel() * t.element_size()
def comm_reset():
    _CNT.clear()
def comm_report(tag):
    parts = "  ".join(f"{k}×{v[0]}（{v[1]/2**20:.1f} MiB）" for k, v in sorted(_CNT.items()))
    print(f"[通信账] {tag}：{parts}", flush=True)
    _CNT.clear()

L, H, T, M = 12, 1024, 256, 4   # H 与 exp10 相同，层数加倍让每段算力占比更高；M = micro-batch 数
HALF = L // 2
lr = 0.01
torch.manual_seed(42)           # 两卡同种子 → 同一份完整权重，各取自己那段
Ws_full = [torch.randn(H, H) / H**0.5 for _ in range(L)]
torch.manual_seed(1000)         # 两卡同种子 → 同一份数据（PP 切的是模型，不是数据）
X = torch.randn(T, H)
Yt = torch.randn(T, H)
Xs = list(X.chunk(M))
Ys = list(Yt.chunk(M))
last_seg = (rank == world - 1)

def gelu_prime(z):
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def make_ws():
    return [w.clone() for w in Ws_full[rank * HALF:(rank + 1) * HALF]]

def fwd(x, Ws):
    """本段前向，返回 (As, Zs)；本段输出 = As[-1]。"""
    A = x
    As, Zs = [A], []
    for i in range(HALF):
        z = A @ Ws[i]
        Zs.append(z)
        A = z if (last_seg and i == HALF - 1) else F.gelu(z)
        As.append(A)
    return As, Zs

def bwd(st, dA, Ws, G):
    """本段反向：梯度累加进 G，返回传给上一段的边界梯度。"""
    As, Zs = st
    for i in reversed(range(HALF)):
        dZ = dA if (last_seg and i == HALF - 1) else dA * gelu_prime(Zs[i])
        G[i] += As[i].T @ dZ
        dA = dZ @ Ws[i].T
    return dA

# ---- 观测工具：recv 空等时间（气泡的实测）与在途激活峰值 ----
IDLE = [0.0]
PEAK = [0]
LIVE = [0]
def hold(st):
    LIVE[0] += 1
    PEAK[0] = max(PEAK[0], LIVE[0])
def drop(st):
    LIVE[0] -= 1

def send_act(t, dst):
    _rec("send", t)
    dist.send(t.contiguous(), dst)

def recv_act(shape, src):
    buf = torch.empty(shape)
    t0 = time.perf_counter()
    dist.recv(buf, src)              # 阻塞在这里的时间 = 等上一段 = 气泡
    IDLE[0] += time.perf_counter() - t0
    _rec("recv", buf)
    return buf

def irecv_wait(shape, src):
    buf = torch.empty(shape)
    req = dist.irecv(buf, src)
    t0 = time.perf_counter()
    req.wait()
    IDLE[0] += time.perf_counter() - t0
    _rec("recv", buf)
    return buf

def update(Ws, G):
    for i in range(HALF):
        Ws[i] -= lr * G[i]           # 各段更新各段的参数，没有梯度同步

# ---------- naive：整个 batch 一次流过（m=1） ----------
def step_naive(Ws):
    G = [torch.zeros_like(w) for w in Ws]
    if rank == 0:
        st = fwd(X, Ws)
        hold(st)
        send_act(st[0][-1], 1)
        dA = recv_act(X.shape, 1)           # 等 rank 1 算完前向+反向
        bwd(st, dA, Ws, G)
        drop(st)
    else:
        act = recv_act(X.shape, 0)      # 等 rank 0 的前向
        st = fwd(act, Ws)
        hold(st)
        dA = 2 * (st[0][-1] - Yt) / st[0][-1].numel()
        dA = bwd(st, dA, Ws, G)
        drop(st)
        send_act(dA, 0)
    update(Ws, G)

# ---------- GPipe：m 份全部前向完，再全部反向 ----------
def step_gpipe(Ws):
    G = [torch.zeros_like(w) for w in Ws]
    states = []
    if rank == 0:
        for j in range(M):              # 前向全部打完
            st = fwd(Xs[j], Ws)
            hold(st)
            states.append(st)
            send_act(st[0][-1], 1)
        for j in range(M):              # 再逐个反向
            dA = recv_act(Xs[j].shape, 1)
            bwd(states[j], dA, Ws, G)
            drop(states[j])
    else:
        for j in range(M):
            act = recv_act(Xs[j].shape, 0)
            st = fwd(act, Ws)
            hold(st)
            states.append(st)
        for j in range(M):
            dA = 2 * (states[j][0][-1] - Ys[j]) / (states[j][0][-1].numel() * M)
            dA = bwd(states[j], dA, Ws, G)
            drop(states[j])
            send_act(dA, 0)
    update(Ws, G)

# ---------- 1F1B：预热 1 份，之后前向反向交替，收尾清尾 ----------
def step_1f1b(Ws):
    G = [torch.zeros_like(w) for w in Ws]
    reqs = []                           # isend 句柄与缓冲，保活到 step 末
    if rank == 0:
        st = fwd(Xs[0], Ws)             # 预热：先灌 1 份
        hold(st)
        states = [st]
        reqs.append((dist.isend(st[0][-1].contiguous(), 1), st[0][-1]))
        _rec("send", st[0][-1])
        for j in range(1, M):           # 稳态：1 次前向 + 1 次反向
            st = fwd(Xs[j], Ws)
            hold(st)
            states.append(st)
            reqs.append((dist.isend(st[0][-1].contiguous(), 1), st[0][-1]))
            _rec("send", st[0][-1])
            dA = irecv_wait(Xs[j].shape, 1)     # 等第 j-1 份的梯度回来
            bwd(states[j - 1], dA, Ws, G)
            drop(states[j - 1])
        dA = irecv_wait(Xs[-1].shape, 1)        # 冷却：清掉最后一份
        bwd(states[-1], dA, Ws, G)
        drop(states[-1])
    else:
        for j in range(M):              # 末段：每收一份，前向反向立刻做完
            act = recv_act(Xs[j].shape, 0)
            st = fwd(act, Ws)
            hold(st)
            dA = 2 * (st[0][-1] - Ys[j]) / (st[0][-1].numel() * M)
            dA = bwd(st, dA, Ws, G)
            drop(st)
            reqs.append((dist.isend(dA.contiguous(), 0), dA))
            _rec("send", dA)
    for r, _ in reqs:
        r.wait()
    update(Ws, G)

def bench(tag, fn, n=5):
    global IDLE, PEAK, LIVE
    Ws = make_ws()
    fn(Ws)                              # 预热（建组等成本不计时）
    comm_reset()
    IDLE, PEAK, LIVE = [0.0], [0], [0]
    ts, idles = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(Ws)
        ts.append((time.perf_counter() - t0) * 1e3)
        idles.append(IDLE[0] * 1e3)
        IDLE[0] = 0.0
    if rank == 0:
        comm_report(f"{tag}（{n} 步合计）")
    print(f"{tag:<6} rank{rank} 每 step：{'  '.join(f'{t:.0f}' for t in ts)} ms"
          f"｜空等 {'  '.join(f'{t:.0f}' for t in idles)} ms"
          f"｜在途激活峰值 {PEAK[0]} 份", flush=True)
    return Ws

p, m = world, M
print(f"rank{rank} 理论气泡率：naive (p-1)/(1+p-1) = {(p-1)/p:.0%}，"
      f"m={m} 时 (p-1)/(m+p-1) = {(p-1)/(m+p-1):.0%}", flush=True)

Ws_a = bench("naive", step_naive)
Ws_g = bench("gpipe", step_gpipe)
Ws_f = bench("1f1b", step_1f1b)

# ---- 正确性：三种调度各走 6 步（上面 bench 的 1 预热 + 5 计时），互相一致即等价 ----
ok = all(torch.allclose(a, b, rtol=1e-4, atol=1e-5) and torch.allclose(a, c, rtol=1e-4, atol=1e-5)
         for a, b, c in zip(Ws_a, Ws_g, Ws_f))
print(f"rank{rank} 三种调度各走 6 步后权重一致：{ok}", flush=True)

dist.destroy_process_group()
