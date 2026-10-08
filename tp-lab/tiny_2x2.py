# tiny_2x2.py —— 2.6 口算实例的验证脚本（CPU / gloo，2 进程）
# 运行：torchrun --nproc_per_node=2 tiny_2x2.py
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank = dist.get_rank()

def log(msg):
    print(f"rank{rank} | {msg}", flush=True)

class F(torch.autograd.Function):          # = _CopyToModelParallelRegion
    @staticmethod
    def forward(ctx, x):
        log("f.forward ：恒等透传 X（不动数据，不通信）")
        return x
    @staticmethod
    def backward(ctx, grad_out):
        log(f"f.backward：手里只有半份 dX = {grad_out.tolist()}，发起 all-reduce！")
        g = grad_out.clone()
        dist.all_reduce(g)                                  # ← 集合通信 ②
        log(f"f.backward：all-reduce 完成，两份求和 = {g.tolist()}")
        return g

class G(torch.autograd.Function):          # = _ReduceFromModelParallelRegion
    @staticmethod
    def forward(ctx, x):
        log(f"g.forward ：手里是部分和 P = {x.tolist()}，发起 all-reduce！")
        y = x.clone()
        dist.all_reduce(y)                                  # ← 集合通信 ①
        log(f"g.forward ：all-reduce 完成，Y = {y.tolist()}")
        return y
    @staticmethod
    def backward(ctx, grad_out):
        log("g.backward ：恒等透传 dY（不通信）")
        return grad_out

# ---- 数据：X[2,2]，W1[2,4] 列切成两个 [2,2]，W2[4,2] 行切成两个 [2,2] ----
X0 = torch.tensor([[1., 2.], [3., 4.]])
W1f = torch.tensor([[1., 0., 2., 1.],
                    [0., 1., 1., 2.]])
W2f = torch.tensor([[1., 0.],
                    [0., 1.],
                    [1., 1.],
                    [2., 0.]])
dY = torch.tensor([[1., 0.], [0., 1.]])   # 上游梯度，取单位阵方便口算

X  = X0.clone().requires_grad_(True)                      # 两卡各持完整 X
W1 = W1f[:, rank*2:(rank+1)*2].clone().requires_grad_(True)  # 卡 i 拿一半列
W2 = W2f[rank*2:(rank+1)*2, :].clone().requires_grad_(True)  # 卡 i 拿一半行

log(f"我持有的权重分片：W1_{rank} = {W1.tolist()}，W2_{rank} = {W2.tolist()}")
dist.barrier()
log("===== 前向开始 =====")
A = F.apply(X)                     # f 守在入口
H = A @ W1
log(f"本地矩阵乘 A@W1_{rank} = {H.tolist()}（零通信）")
P = H @ W2
log(f"本地矩阵乘 H@W2_{rank} = 部分和 {P.tolist()}（零通信）")
Y = G.apply(P)                     # g 守在出口
log("===== 前向结束 =====")
dist.barrier()
log("===== 反向开始 =====")
Y.backward(dY)
log(f"===== 反向结束，X.grad = {X.grad.tolist()} =====")

if rank == 0:
    Xr = X0.clone().requires_grad_(True)
    Yr = Xr @ W1f @ W2f            # 单卡参考
    Yr.backward(dY)
    print(f"\n[单卡参考] Y = {Yr.tolist()}，dX = {Xr.grad.tolist()}", flush=True)

dist.destroy_process_group()
