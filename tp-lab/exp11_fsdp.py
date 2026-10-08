# exp11_fsdp.py —— 实验二：mini-FSDP——参数 all-gather、梯度 reduce-scatter，与单卡对比
# 每张卡只存 1/t 的参数分片；使用时 all-gather 拼全（用完即释放），
# 梯度 reduce-scatter 求和、只留本卡那一片。两卡使用不同数据 —— 这是 DP 的本义。
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import lab_init, _rec, comm_reset, comm_report

dev, rank, world = lab_init("实验二：mini-FSDP")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

torch.manual_seed(0)                     # 两卡同种子 → 同一份"完整权重"
H, T = 1024, 256
W1_full = torch.randn(H, 4 * H, device=dev) / H**0.5
W2_full = torch.randn(4 * H, H, device=dev) / (4 * H) ** 0.5
torch.manual_seed(1000 + rank)           # 两卡不同数据 —— 真正的 DP
X = torch.randn(T, H, device=dev)
target = torch.randn(T, H, device=dev)

class _FsdpMatmul(torch.autograd.Function):
    """mini-FSDP 的一次矩阵乘：参数按行分片存储，使用时 all-gather 拼全；
    反向重新拼一次参数，梯度 reduce-scatter 求和、只留本卡那一片。"""

    @staticmethod
    def forward(ctx, x, w_shard):
        world = dist.get_world_size()
        _rec("all_gather", w_shard)
        outs = [torch.empty_like(w_shard) for _ in range(world)]
        dist.all_gather(outs, w_shard.contiguous())
        w_full = torch.cat(outs, dim=0)      # 拼全参数（用完即释放，不保留）
        ctx.save_for_backward(x, w_shard)    # ctx 只保存分片，不保存全量
        return x @ w_full

    @staticmethod
    def backward(ctx, grad_output):
        x, w_shard = ctx.saved_tensors
        world = dist.get_world_size()
        _rec("all_gather", w_shard)
        outs = [torch.empty_like(w_shard) for _ in range(world)]
        dist.all_gather(outs, w_shard.contiguous())
        w_full = torch.cat(outs, dim=0)      # 参数再拼一次 —— 前向没保存，只能重拼
        grad_x = grad_output @ w_full.T
        grad_w_full = x.T @ grad_output      # 本卡数据的完整梯度……
        out = torch.empty_like(w_shard)
        _rec("reduce_scatter", out)
        dist.reduce_scatter_tensor(out, grad_w_full.contiguous())  # ……求和后只留本卡那一片
        return grad_x, out

def fsdp_matmul(x, w_shard): return _FsdpMatmul.apply(x, w_shard)

def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-4)

# ---- 参考 loss：本卡数据 + 完整权重，等价于单卡计算 ----
Xr = X.clone().requires_grad_(True)
W1r = W1_full.clone().requires_grad_(True)
W2r = W2_full.clone().requires_grad_(True)
loss_ref = ((F.gelu(Xr @ W1r) @ W2r - target) ** 2).mean()

# ---- 参考梯度：两卡数据的梯度之和（DP 的求和语义；训练时的 ÷world 这里省略） ----
g1 = torch.zeros_like(W1_full)
g2 = torch.zeros_like(W2_full)
for j in range(world):
    torch.manual_seed(1000 + j)          # 复现第 j 张卡的数据
    Xj = torch.randn(T, H, device=dev)
    Yj = torch.randn(T, H, device=dev)
    W1j = W1_full.clone().requires_grad_(True)
    W2j = W2_full.clone().requires_grad_(True)
    ((F.gelu(Xj @ W1j) @ W2j - Yj) ** 2).mean().backward()
    g1 += W1j.grad
    g2 += W2j.grad

# ---- FSDP：每卡只持有 1/world 的参数分片（沿行切，便于梯度分片） ----
rows1, rows2 = H // world, (4 * H) // world
W1 = W1_full[rank * rows1:(rank + 1) * rows1].clone().requires_grad_(True)
W2 = W2_full[rank * rows2:(rank + 1) * rows2].clone().requires_grad_(True)

shard_b = (W1.numel() + W2.numel()) * 4
full_b = (W1_full.numel() + W2_full.numel()) * 4
if rank == 0:
    print(f"本卡参数 {shard_b / 2**20:.1f} MB（完整版 {full_b / 2**20:.1f} MB，已 ÷{world}）",
          flush=True)

comm_reset()
out = fsdp_matmul(F.gelu(fsdp_matmul(X, W1)), W2)
loss = ((out - target) ** 2).mean()
comm_report("forward")
comm_reset()                         # 清零：让 backward 的通信单独统计
loss.backward()
comm_report("backward")

ok_loss = same(loss.detach(), loss_ref.detach())
ok_w1 = same(W1.grad, g1[rank * rows1:(rank + 1) * rows1])
ok_w2 = same(W2.grad, g2[rank * rows2:(rank + 1) * rows2])
status = "OK" if (ok_loss and ok_w1 and ok_w2) else "FAIL"
print(f"rank {rank}: loss 对比 {ok_loss}，dW1 分片对比 {ok_w1}，dW2 分片对比 {ok_w2} "
      f"[{status}]", flush=True)
dist.destroy_process_group()