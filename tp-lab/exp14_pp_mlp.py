# exp14_pp_mlp.py —— 实验一：手写 2 段流水线——send 激活、recv 梯度，各段更新各段的参数
# rank 0 拿前 3 层，rank 1 拿后 3 层；前向 send 边界激活，反向 send 边界梯度。
# 注意整个 step 里没有 all_reduce：两段参数不共享，各更新各的——这是 PP 与 DP 的本质区别。
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import lab_init, _rec, comm_reset, comm_report

dev, rank, world = lab_init("实验一：手写 2 段流水线")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

L, H, T = 6, 1024, 256          # 与 exp10 同尺寸，方便对照
HALF = L // 2                   # 每段 3 层
lr = 0.01

torch.manual_seed(42)           # 两卡同种子 → 同一份完整权重，各取自己那段
Ws_full = [torch.randn(H, H, device=dev) / H**0.5 for _ in range(L)]
Ws = [w.clone() for w in Ws_full[rank * HALF:(rank + 1) * HALF]]
torch.manual_seed(1000)         # 两卡同种子 → 同一份数据（PP 里数据相同，切的是模型）
X = torch.randn(T, H, device=dev)
Yt = torch.randn(T, H, device=dev)

def gelu_prime(z):
    # gelu(x) = x·Φ(x)；导数 = Φ(x) + x·φ(x)
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def fwd_seg(x, last_seg):
    """本段前向：返回 (As, Zs, 输出)。只有全模型最后一层不过 gelu。"""
    A = x
    As, Zs = [A], []
    for i in range(HALF):
        z = A @ Ws[i]
        Zs.append(z)
        A = z if (last_seg and i == HALF - 1) else F.gelu(z)
        As.append(A)
    return As, Zs, A

def bwd_seg(As, Zs, dA, last_seg):
    """本段反向：更新本段参数，返回传给上一段的边界梯度。"""
    for i in reversed(range(HALF)):
        dZ = dA if (last_seg and i == HALF - 1) else dA * gelu_prime(Zs[i])
        dW = As[i].T @ dZ
        dA = dZ @ Ws[i].T                  # 先用旧权重算上传的梯度……
        Ws[i] -= lr * dW                   # ……再就地更新本段参数——没有任何梯度同步
    return dA

comm_reset()
if rank == 0:
    As, Zs, act = fwd_seg(X, last_seg=False)
    _rec("send", act)
    dist.send(act.contiguous(), dst=1)     # 前向：把边界激活寄给下一段
    dA = torch.empty(T, H, device=dev)
    _rec("recv", dA)
    dist.recv(dA, src=1)                   # 反向：等下一段把边界梯度寄回来
    bwd_seg(As, Zs, dA, last_seg=False)
    loss = None                            # rank 0 不算 loss
else:
    act = torch.empty(T, H, device=dev)
    _rec("recv", act)
    dist.recv(act, src=0)                  # 前向：收上一段寄来的激活
    As, Zs, out = fwd_seg(act, last_seg=True)
    loss = ((out - Yt) ** 2).mean()
    dA = 2 * (out - Yt) / out.numel()      # MSE 的导数
    dA = bwd_seg(As, Zs, dA, last_seg=True)
    _rec("send", dA)
    dist.send(dA.contiguous(), dst=0)      # 反向：把边界梯度寄回上一段

comm_report("1 个 step")

# ---- 正确性：与单卡完整模型对拍（两卡各自在本地算同一份参考） ----
Wr = [w.clone() for w in Ws_full]
As, Zs = [X], []
A = X
for i in range(L):
    z = A @ Wr[i]
    Zs.append(z)
    A = z if i == L - 1 else F.gelu(z)
    As.append(A)
loss_ref = ((As[-1] - Yt) ** 2).mean()
dA = 2 * (As[-1] - Yt) / As[-1].numel()
for i in reversed(range(L)):
    dZ = dA if i == L - 1 else dA * gelu_prime(Zs[i])
    dW = As[i].T @ dZ
    dA = dZ @ Wr[i].T                      # 先用旧权重算上传的梯度
    Wr[i] -= lr * dW

ok_w = all(torch.allclose(a, b, rtol=1e-4, atol=1e-4)
           for a, b in zip(Ws, Wr[rank * HALF:(rank + 1) * HALF]))
if rank == 1:
    ok_loss = torch.allclose(loss, loss_ref, rtol=1e-4, atol=1e-6)
    print(f"rank 1: loss 对比 {ok_loss}，本段 3 层权重对比 {ok_w} "
          f"[{'OK' if ok_loss and ok_w else 'FAIL'}]", flush=True)
else:
    print(f"rank 0: 本段 3 层权重对比 {ok_w} [{'OK' if ok_w else 'FAIL'}]", flush=True)
    full_mb = L * H * H * 4 / 2**20
    print(f"本段参数 {full_mb / 2:.0f} MB（全模型 {full_mb:.0f} MB，天然 ÷{world}）", flush=True)

dist.destroy_process_group()
