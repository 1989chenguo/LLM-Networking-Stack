# exp10_dp_overlap.py —— 实验一：DP 梯度同步——朴素阻塞 vs 分桶异步重叠
# 同一笔通信账，两种付法：backward 算完再整块同步（暴露），
# 还是梯度一就绪就异步投递、藏进后续层的反向计算身后（重叠）。
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import lab_init, _rec, comm_reset, comm_report

dev, rank, world = lab_init("实验一：把通信藏进 backward 身后")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

torch.manual_seed(0)
L, H, T = 6, 1024, 256          # 层数、宽度、token 数
lr = 0.01
X = torch.randn(T, H, device=dev)     # 两卡数据相同（本实验只对通信计时；
Y = torch.randn(T, H, device=dev)     # DP 的"不同数据"语义在 exp11 里对拍）

def make_model():
    torch.manual_seed(42)       # 两种模式从同一份权重出发，最后互相对拍
    Ws = [torch.randn(H, H, device=dev) / H**0.5 for _ in range(L)]
    bs = [torch.randn(H, device=dev) * 0.02 for _ in range(L)]
    return Ws, bs

def gelu_prime(z):
    # gelu(x) = x·Φ(x)；导数 = Φ(x) + x·φ(x)
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def forward(Ws, bs):
    A = X
    As, Zs = [A], []
    for i in range(L):
        z = A @ Ws[i] + bs[i]
        Zs.append(z)
        A = z if i == L - 1 else F.gelu(z)
        As.append(A)
    return As, Zs

def update(Ws, bs, grads):
    for i in range(L):
        n = Ws[i].numel()
        Ws[i] -= lr * grads[i][:n].view_as(Ws[i])
        bs[i] -= lr * grads[i][n:]

def step_naive(Ws, bs):
    """朴素版：整个反向先算完，再逐层阻塞同步 —— 通信整块暴露在身后。"""
    As, Zs = forward(Ws, bs)
    loss = ((As[-1] - Y) ** 2).mean()
    dA = 2 * (As[-1] - Y) / As[-1].numel()   # MSE 的导数
    grads = [None] * L
    for i in reversed(range(L)):             # 反向：梯度按层倒序就绪
        dZ = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = torch.cat([(As[i].T @ dZ).reshape(-1),
                              dZ.sum(0).reshape(-1)]).contiguous()
        dA = dZ @ Ws[i].T
    for i in range(L):                       # ← 通信从这里才开始，且逐层等完
        _rec("all_reduce", grads[i])
        dist.all_reduce(grads[i])            # 阻塞：投单 + 收货一次完成
    update(Ws, bs, grads)
    return loss

def step_overlapped(Ws, bs):
    """重叠版：每一层的梯度一算出来就立刻异步投递，回头继续算上一层。"""
    As, Zs = forward(Ws, bs)
    loss = ((As[-1] - Y) ** 2).mean()
    dA = 2 * (As[-1] - Y) / As[-1].numel()
    grads, handles = [None] * L, []
    for i in reversed(range(L)):
        dZ = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = torch.cat([(As[i].T @ dZ).reshape(-1),
                              dZ.sum(0).reshape(-1)]).contiguous()
        _rec("all_reduce", grads[i])
        handles.append(dist.all_reduce(grads[i], async_op=True))  # ← 投单就走
        dA = dZ @ Ws[i].T                    # 继续算上一层（通信在身后飞）
    for h in handles:
        h.wait()                             # 全部投完，统一收回执
    update(Ws, bs, grads)
    return loss

def time_it(fn, Ws, bs, n=5):
    fn(Ws, bs)                               # 预热：建组成本花在计时外
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(Ws, bs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts

Ws_n, bs_n = make_model()
Ws_o, bs_o = make_model()

comm_reset()
naive_ms = time_it(step_naive, Ws_n, bs_n)
comm_report("naive")

comm_reset()
ovl_ms = time_it(step_overlapped, Ws_o, bs_o)
comm_report("overlapped")

if rank == 0:
    for name, ts in [("naive", naive_ms), ("overlapped", ovl_ms)]:
        steps = "  ".join(f"{t:.1f}" for t in ts)
        print(f"{name:<11} 每 step：{steps} ms", flush=True)
    ok = all(torch.allclose(a, b, rtol=1e-4, atol=1e-5)
             for a, b in zip(Ws_n + bs_n, Ws_o + bs_o))
    print(f"两种走法训 5 步后权重对拍：{ok}（账一样、时间不一样）", flush=True)

if os.environ.get("TRACE"):                  # TRACE=1 导出 profiler 时间线（GPU 上看得最清楚）
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step_overlapped(Ws_o, bs_o)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    prof.export_chrome_trace(f"trace_rank{rank}.json")
    if rank == 0:
        print("已导出 trace_rank0.json：chrome://tracing 打开，找通信 kernel 藏在哪里", flush=True)

dist.destroy_process_group()
