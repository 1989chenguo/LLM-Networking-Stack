# exp7_tp_mlp.py —— 实验一：两卡手写 TP 版 MLP，和"单卡"逐位对拍
import torch
import torch.distributed as dist
from mini_tp import (lab_init, copy_to_tp_region, reduce_from_tp_region,
                     comm_reset, comm_report)

dev, rank, world = lab_init("实验一：手写 TP 版 MLP")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

torch.manual_seed(0)                    # 两卡同种子 → 造出同一份"完整权重"
H, T = 4096, 64                         # hidden 维、token 数（s×b 摊平）
X = torch.randn(T, H, device=dev)
W1_full = torch.randn(H, 4 * H, device=dev) / H**0.5        # column 切
W2_full = torch.randn(4 * H, H, device=dev) / (4 * H)**0.5  # row 切
gY = torch.randn(T, H, device=dev)      # 冒充"上层传下来的梯度"，两卡相同

def same(a, b):                         # fp32 求和顺序不同，容差内一致即对拍成功
    return torch.allclose(a, b, rtol=1e-4, atol=1e-4)

# ---- 参考：每卡独立算完整版，等价把单卡搬上了卡 ----
Xr = X.clone().requires_grad_(True)
W1r = W1_full.clone().requires_grad_(True)
W2r = W2_full.clone().requires_grad_(True)
Y_ref = torch.nn.functional.gelu(Xr @ W1r) @ W2r
Y_ref.backward(gY)

# ---- TP：各卡只持有并计算一半权重 ----
half = (4 * H) // world
X_tp = X.clone().requires_grad_(True)
W1 = W1_full[:, rank * half:(rank + 1) * half].clone().requires_grad_(True)  # 按列切
W2 = W2_full[rank * half:(rank + 1) * half, :].clone().requires_grad_(True)  # 按行切

comm_reset()
A = copy_to_tp_region(X_tp)                     # f：前向恒等，反向才通信
H_loc = torch.nn.functional.gelu(A @ W1)        # [T, 2H]：本地半个矩阵乘，零通信
Y_tp = reduce_from_tp_region(H_loc @ W2)        # g：前向 all-reduce
comm_report("forward")
ok_fwd = same(Y_tp, Y_ref)

Y_tp.backward(gY)
comm_report("backward")
ok_dx = same(X_tp.grad, Xr.grad)
ok_w1 = same(W1.grad, W1r.grad[:, rank * half:(rank + 1) * half])   # 梯度只比
ok_w2 = same(W2.grad, W2r.grad[rank * half:(rank + 1) * half, :])   # 自己那片

status = "OK" if all([ok_fwd, ok_dx, ok_w1, ok_w2]) else "FAIL"
print(f"rank {rank}: forward 对拍 {ok_fwd}，dX 对拍 {ok_dx}，"
      f"dW 分片对拍 {ok_w1 and ok_w2} [{status}]", flush=True)
dist.destroy_process_group()
