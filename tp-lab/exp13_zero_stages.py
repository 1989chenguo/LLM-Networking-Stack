# exp13_zero_stages.py —— 实验四：ZeRO 三档与 DP 对拍——显存、通信量、时间各是多少？
# 同一个 6 层 MLP、同一份初始权重、同样的数据、同一条 mini-Adam 更新规则，四组对拍：
#   dp    ：全量参数 + 全量梯度 + 全量优化器状态；梯度逐层异步 all_reduce（exp10 的做法）
#   zero1 ：参数全量常驻，优化器状态 ÷t；反向零通信，反向后梯度 reduce_scatter 留片，
#           各卡只更新自己的分片，再 all_gather 把参数拼回全量（通信与 DP 同为 2P）
#   zero2 ：在 zero1 基础上把梯度 reduce_scatter 挪进反向逐层异步发起——
#           全量梯度从"攒到反向结束"变成"单层瞬时"（通信不变，梯度显存 ÷t）
#   fsdp  ：参数也 ÷t，用时 all_gather 拼全、用完即释放（exp12 的做法）
# 看三组数字：反向结束时本卡实际持有字节、通信量（计数器）、每 step 耗时。
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

L, H, T = 6, 1024, 256          # 与 exp10/12 同尺寸，方便对照
lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
torch.manual_seed(1000 + rank)  # 两卡不同数据 —— 真正的 DP
X = torch.randn(T, H)
Yt = torch.randn(T, H)

def gelu_prime(z):
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def make_full():
    torch.manual_seed(42)       # 四组实验从同一份权重出发
    return [torch.randn(H, H) / H**0.5 for _ in range(L)]

r = H // world                  # 每卡分片的行数
def shard(W):
    return W[rank * r:(rank + 1) * r].clone()

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
    return 2 * (As[-1] - Yt) / As[-1].numel()

def adam_(p, g, m, v, t):       # 就地 mini-Adam：四组共用同一条更新规则，结果才能互校
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    p.add_(mh / (vh.sqrt() + eps), alpha=-lr)

# ---- 显存账：反向结束时本卡实际持有的张量（参数 / 优化器状态 / 梯度），只记一次 ----
_PEAK = []
def note_peak(*groups):
    if not _PEAK:                        # 立即折算成字节快照，后续释放不影响
        _PEAK.extend(sum(t.numel() * t.element_size() for t in g) for g in groups)
def peak_report(tag):
    names = ("参数", "优化器状态", "梯度")
    detail = " + ".join(f"{n} {b / 2**20:.0f}" for n, b in zip(names, _PEAK))
    if rank == 0:
        print(f"[显存账] {tag}：反向结束时本卡持有 {sum(_PEAK) / 2**20:.0f} MiB"
              f"（{detail} MiB）", flush=True)

def gather_async(sw):
    outs = [torch.empty_like(sw) for _ in range(world)]
    h = dist.all_gather(outs, sw.contiguous(), async_op=True)
    return outs, h

# ---------- dp：全量参数、全量梯度、全量优化器状态 ----------
def step_dp(st, t):
    W, m, v = st
    As, Zs = forward(W)
    dA = dA_loss(As)
    G, hs = [None] * L, []
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        G[i] = As[i].T @ dz
        _rec("all_reduce", G[i])
        hs.append(dist.all_reduce(G[i], async_op=True))
        dA = dz @ W[i].T
    note_peak(W, m + v, G)                       # 全量梯度一直攒到反向结束
    for h in hs:
        h.wait()
    for i in range(L):
        adam_(W[i], G[i], m[i], v[i], t)         # 更新全量参数（优化器状态也是全量）

# ---------- zero1：参数全量，优化器状态 ÷t；反向零通信 ----------
def step_zero1(st, t):
    W, m, v = st
    As, Zs = forward(W)
    dA = dA_loss(As)
    G = [None] * L
    for i in reversed(range(L)):                 # 反向：参数在手边，没有任何通信
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        G[i] = As[i].T @ dz
        dA = dz @ W[i].T
    note_peak(W, m + v, G)                       # 全量梯度攒到反向结束
    for i in range(L):
        gsh = torch.empty(r, H)
        _rec("reduce_scatter", gsh)
        dist.reduce_scatter_tensor(gsh, G[i].contiguous())  # 求和 + 只留本卡分片
        G[i] = None                              # 全量梯度用完即扔
        sl = W[i][rank * r:(rank + 1) * r]
        adam_(sl, gsh, m[i], v[i], t)            # 只更新自己负责的那 1/t
        outs = [torch.empty(r, H) for _ in range(world)]
        _rec("all_gather", outs[rank])
        dist.all_gather(outs, sl.contiguous())   # 把更新后的分片拼回全量参数
        W[i].copy_(torch.cat(outs, dim=0))

# ---------- zero2：在 zero1 基础上把梯度规约挪进反向，逐层异步 ----------
def step_zero2(st, t):
    W, m, v = st
    As, Zs = forward(W)
    dA = dA_loss(As)
    gsh, hs = [None] * L, []
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz                        # 本层全量梯度，循环体结束即释放
        gsh[i] = torch.empty(r, H)
        _rec("reduce_scatter", gsh[i])
        hs.append(dist.reduce_scatter_tensor(gsh[i], gw.contiguous(), async_op=True))
        dA = dz @ W[i].T                         # 通信与上一层的计算并行
    note_peak(W, m + v, gsh)                     # 梯度常驻只剩分片
    for h in hs:
        h.wait()
    for i in range(L):
        sl = W[i][rank * r:(rank + 1) * r]
        adam_(sl, gsh[i], m[i], v[i], t)
        outs = [torch.empty(r, H) for _ in range(world)]
        _rec("all_gather", outs[rank])
        dist.all_gather(outs, sl.contiguous())
        W[i].copy_(torch.cat(outs, dim=0))

# ---------- fsdp：参数也 ÷t，用时拼全、用完即释放 ----------
def step_fsdp(st, t):
    SW, m, v = st
    outs, h = gather_async(SW[0])
    _rec("all_gather", SW[0])
    A = X
    As, Zs = [A], []
    for i in range(L):                           # 前向：预取下一层参数
        h.wait()
        w = torch.cat(outs, dim=0)
        if i + 1 < L:
            outs, h = gather_async(SW[i + 1])
            _rec("all_gather", SW[i + 1])
        z = A @ w
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    dA = dA_loss(As)
    outs, h = gather_async(SW[L - 1])
    _rec("all_gather", SW[L - 1])
    gsh, rhs = [None] * L, []
    for i in reversed(range(L)):                 # 反向：预取上一层参数
        h.wait()
        w = torch.cat(outs, dim=0)
        if i - 1 >= 0:
            outs, h = gather_async(SW[i - 1])
            _rec("all_gather", SW[i - 1])
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz
        gsh[i] = torch.empty_like(SW[i])
        _rec("reduce_scatter", gsh[i])
        rhs.append(dist.reduce_scatter_tensor(gsh[i], gw.contiguous(), async_op=True))
        dA = dz @ w.T
    note_peak(SW, m + v, gsh)                    # 参数、梯度、优化器状态全是分片
    for h_ in rhs:
        h_.wait()
    for i in range(L):
        adam_(SW[i], gsh[i], m[i], v[i], t)      # 更新分片即终态，无需拼回

def bench(tag, step_fn, mk_state, n=5):
    global _PEAK
    _PEAK = []
    st = mk_state()
    step_fn(st, 1)                               # 预热（建组等成本不计时），同时记下显存账
    peak_report(tag)
    comm_reset()
    ts = []
    for k in range(2, n + 2):
        t0 = time.perf_counter()
        step_fn(st, k)
        ts.append((time.perf_counter() - t0) * 1e3)
    if rank == 0:
        comm_report(f"{tag}（{n} 步合计）")
        print(f"{tag:<7} 每 step：{'  '.join(f'{x:.1f}' for x in ts)} ms", flush=True)

def mk_dp():    return make_full(), [torch.zeros(H, H) for _ in range(L)], [torch.zeros(H, H) for _ in range(L)]
def mk_z1():    return make_full(), [torch.zeros(r, H) for _ in range(L)], [torch.zeros(r, H) for _ in range(L)]
def mk_fsdp():  return [shard(w) for w in make_full()], [torch.zeros(r, H) for _ in range(L)], [torch.zeros(r, H) for _ in range(L)]

bench("dp",     step_dp,    mk_dp)
bench("zero1",  step_zero1, mk_z1)
bench("zero2",  step_zero2, mk_z1)
bench("fsdp",   step_fsdp,  mk_fsdp)

# 正确性：四组从同一份初始权重出发各走 1 步，FSDP 拼回分片，互相校验
st_dp, st_z1, st_z2, st_f = mk_dp(), mk_z1(), mk_z1(), mk_fsdp()
step_dp(st_dp, 1); step_zero1(st_z1, 1); step_zero2(st_z2, 1); step_fsdp(st_f, 1)
ok = True
for i in range(L):
    outs = [torch.empty_like(st_f[0][i]) for _ in range(world)]
    dist.all_gather(outs, st_f[0][i].contiguous())
    wf = torch.cat(outs, dim=0)
    ok = ok and torch.allclose(wf, st_dp[0][i], rtol=1e-4, atol=1e-5) \
             and torch.allclose(wf, st_z1[0][i], rtol=1e-4, atol=1e-5) \
             and torch.allclose(wf, st_z2[0][i], rtol=1e-4, atol=1e-5)
if rank == 0:
    print(f"四种方式各走 1 步后权重一致：{ok}", flush=True)

dist.destroy_process_group()