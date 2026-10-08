# exp12_dp_vs_fsdp.py —— 实验三：纯 DP vs FSDP——用通信换显存，训练时间变多少？
# 同一个 6 层 MLP、同一份初始权重、同样的数据，四组对拍：
#   DP   朴素：backward 全部算完，再逐层阻塞 all_reduce（通信整块暴露）
#   DP   重叠：逐层梯度一算出来就 async all_reduce（exp10 的做法）
#   FSDP 朴素：每层阻塞 all-gather 拼参数 + 阻塞 reduce-scatter 留切片
#   FSDP 重叠：前向/反向预取下一层参数，梯度规约异步发起（真实 FSDP 的做法）
# 看两组数字：通信量（计数器，恒定的 1.5 倍）与每 step 耗时（计时，因机器而异）。
import time
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

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

L, H, T = 6, 1024, 256          # 与 exp10 同尺寸，方便对照
lr = 0.01
torch.manual_seed(1000 + rank)  # 两卡不同数据 —— 真正的 DP
X = torch.randn(T, H)
Ytarget = torch.randn(T, H)

def gelu_prime(z):
    # gelu(x) = x·Φ(x)；导数 = Φ(x) + x·φ(x)
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def make_full():
    torch.manual_seed(42)       # 四组实验从同一份权重出发
    return [torch.randn(H, H, device="cpu") / H**0.5 for _ in range(L)]

def shard(W):                   # 沿行切，返回本卡分片（FSDP 的常驻参数）
    r = H // world
    return W[rank * r:(rank + 1) * r].clone()

def gather_async(sw):           # 异步 all-gather：投出就去干别的，句柄稍等
    outs = [torch.empty_like(sw) for _ in range(world)]
    h = dist.all_gather(outs, sw.contiguous(), async_op=True)
    return outs, h

def gather(sw):                 # 阻塞 all-gather：拼回完整参数
    outs = [torch.empty_like(sw) for _ in range(world)]
    _rec("all_gather", sw)
    dist.all_gather(outs, sw.contiguous())
    return torch.cat(outs, dim=0)

def forward(Ws):
    A = X
    As, Zs = [A], []
    for i in range(L):
        z = A @ Ws[i]
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    return As, Zs

def dA_loss(As):
    return 2 * (As[-1] - Ytarget) / As[-1].numel()   # MSE 的导数

# ---------- 纯 DP：每卡常驻全量参数，梯度 all_reduce 求和 ----------

def dp_naive(Ws):
    """朴素版：整个反向算完，再逐层阻塞 all_reduce —— 通信整块暴露。"""
    As, Zs = forward(Ws)
    dA = dA_loss(As)
    grads = [None] * L
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = As[i].T @ dz
        dA = dz @ Ws[i].T
    for i in range(L):
        _rec("all_reduce", grads[i])
        dist.all_reduce(grads[i])            # 阻塞：发起并等完
    for i in range(L):
        Ws[i] -= lr * grads[i]

def dp_overlap(Ws):
    """重叠版：每层的梯度一算出来就立刻异步 all_reduce，回头继续算上一层。"""
    As, Zs = forward(Ws)
    dA = dA_loss(As)
    grads = [None] * L
    hs = []
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = As[i].T @ dz
        _rec("all_reduce", grads[i])
        hs.append(dist.all_reduce(grads[i], async_op=True))  # 投单就走
        dA = dz @ Ws[i].T                    # 通信在身后飞
    for h in hs:
        h.wait()
    for i in range(L):
        Ws[i] -= lr * grads[i]

# ---------- FSDP：每卡常驻 1/t 参数分片，用到拼全、梯度求和留切片 ----------

def fsdp_naive(SWs):
    """朴素版：每层阻塞 all-gather 拼参数（前向一次、反向一次），阻塞 reduce-scatter。"""
    Wf = [gather(s) for s in SWs]            # 前向：逐层阻塞拼全
    As, Zs = forward(Wf)
    dA = dA_loss(As)
    gsh = [None] * L
    for i in reversed(range(L)):
        w = gather(SWs[i])                   # 反向：前向没留全量，再拼一次
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz                    # 本卡数据的完整梯度
        out = torch.empty_like(SWs[i])
        _rec("reduce_scatter", out)
        dist.reduce_scatter_tensor(out, gw.contiguous())   # 求和 + 只留本卡那片
        gsh[i] = out
        dA = dz @ w.T
    for i in range(L):
        SWs[i] -= lr * gsh[i]

def fsdp_overlap(SWs):
    """重叠版：算第 i 层时，第 i+1 层（反向则是第 i-1 层）的参数已在路上；
    梯度 reduce-scatter 异步发起，继续算上一层。"""
    outs, h = gather_async(SWs[0])
    _rec("all_gather", SWs[0])
    A = X
    As, Zs = [A], []
    for i in range(L):                       # 前向：预取下一层参数
        h.wait()
        w_full = torch.cat(outs, dim=0)
        if i + 1 < L:
            outs, h = gather_async(SWs[i + 1])
            _rec("all_gather", SWs[i + 1])
        z = A @ w_full
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    dA = dA_loss(As)
    outs, h = gather_async(SWs[L - 1])
    _rec("all_gather", SWs[L - 1])
    gsh = [None] * L
    rhs = []
    for i in reversed(range(L)):             # 反向：预取上一层参数
        h.wait()
        w_full = torch.cat(outs, dim=0)
        if i - 1 >= 0:
            outs, h = gather_async(SWs[i - 1])
            _rec("all_gather", SWs[i - 1])
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz
        out = torch.empty_like(SWs[i])
        _rec("reduce_scatter", out)
        rhs.append(dist.reduce_scatter_tensor(out, gw.contiguous(), async_op=True))
        gsh[i] = out
        dA = dz @ w_full.T
    for h_ in rhs:
        h_.wait()
    for i in range(L):
        SWs[i] -= lr * gsh[i]

def bench(name, fn, mk, n=5):
    """预热 1 步（不计时）+ 计时 n 步；计数器清零后只统计这 n 步。"""
    W = mk()
    fn(W)                                    # 预热：建组等成本花在计时外
    comm_reset()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(W)
        ts.append((time.perf_counter() - t0) * 1e3)
    comm_report(f"{name}（{n} 步合计）")
    print(f"{name:<13} 每 step：{'  '.join(f'{t:.1f}' for t in ts)} ms", flush=True)
    return W

if rank == 0:
    dp_mb = L * H * H * 4 / 2**20
    print(f"常驻参数：DP 每卡 {dp_mb:.0f} MB，FSDP 每卡 {dp_mb // world:.0f} MB（已 ÷{world}）", flush=True)

bench("dp_naive", dp_naive, make_full)
bench("dp_overlap", dp_overlap, make_full)
bench("fsdp_naive", fsdp_naive, lambda: [shard(w) for w in make_full()])
bench("fsdp_overlap", fsdp_overlap, lambda: [shard(w) for w in make_full()])

# 正确性：从同一份初始权重出发，DP 与 FSDP 各走 1 步，拼回分片对比
Ws = make_full()
SWs = [shard(w) for w in make_full()]
dp_overlap(Ws)
fsdp_overlap(SWs)
ok = True
for i in range(L):
    outs = [torch.empty_like(SWs[i]) for _ in range(world)]
    dist.all_gather(outs, SWs[i].contiguous())
    ok = ok and torch.allclose(torch.cat(outs, dim=0), Ws[i], rtol=1e-4, atol=1e-5)
if rank == 0:
    print(f"DP 与 FSDP 各走 1 步后权重一致：{ok}", flush=True)

dist.destroy_process_group()