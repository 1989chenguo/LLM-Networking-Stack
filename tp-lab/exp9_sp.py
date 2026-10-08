# exp9_sp.py —— 实验三：序列并行版同一层——次数翻倍，字节不变
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import (lab_init, gather_from_seq_region, reduce_scatter_to_seq_region,
                     comm_reset, comm_report)

dev, rank, world = lab_init("实验三：序列并行")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

torch.manual_seed(0)
S, B, H, NH = 256, 2, 2048, 8
HD = H // NH
X_full = torch.randn(S, B, H, device=dev)
Att_f = [torch.randn(H, H, device=dev) / H**0.5 for _ in range(4)]
W1f = torch.randn(H, 4 * H, device=dev) / H**0.5
W2f = torch.randn(4 * H, H, device=dev) / (4 * H)**0.5

def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-4)

def ln(x):                       # 无参数 LayerNorm：逐 token 归一化，沿序列切免费
    return F.layer_norm(x, (x.size(-1),))

def attn(x, wq, wk, wv, wo):
    q, k, v = x @ wq, x @ wk, x @ wv
    nh = q.size(-1) // HD
    q = q.view(S, B, nh, HD).permute(1, 2, 0, 3)
    k = k.view(S, B, nh, HD).permute(1, 2, 0, 3)
    v = v.view(S, B, nh, HD).permute(1, 2, 0, 3)
    o = F.scaled_dot_product_attention(q, k, v)
    return o.permute(2, 0, 1, 3).reshape(S, B, nh * HD) @ wo

def mlp(x, w1, w2):
    return F.gelu(x @ w1) @ w2

# ---- 参考：完整序列 + 完整权重，等价单卡 ----
Xr = X_full.clone().requires_grad_(True)
Attr = [w.clone().requires_grad_(True) for w in Att_f]
W1r, W2r = W1f.clone().requires_grad_(True), W2f.clone().requires_grad_(True)
h1 = Xr + attn(ln(Xr), *Attr)
Y_ref = h1 + mlp(ln(h1), W1r, W2r)
gY = torch.randn(S, B, H, device=dev)
Y_ref.backward(gY)

# ---- SP：序列沿 dim 0 切开，权重切法与实验二相同 ----
sub, half = H // world, (4 * H) // world
sl = slice(rank * S // world, (rank + 1) * S // world)
Wq, Wk, Wv = [Att_f[i][:, rank*sub:(rank+1)*sub].clone().requires_grad_(True)
              for i in range(3)]
Wo = Att_f[3][rank*sub:(rank+1)*sub, :].clone().requires_grad_(True)
W1 = W1f[:, rank*half:(rank+1)*half].clone().requires_grad_(True)
W2 = W2f[rank*half:(rank+1)*half, :].clone().requires_grad_(True)
X_loc = X_full[sl].clone().requires_grad_(True)          # 每卡只有 [S/2, B, H]

comm_reset()
a = gather_from_seq_region(ln(X_loc))                    # f'：all-gather → [S,B,H]
h_attn = reduce_scatter_to_seq_region(attn(a, Wq, Wk, Wv, Wo))  # g'：→ [S/2,B,H]
h1 = X_loc + h_attn                                      # 残差相加：分片上逐元素
m = gather_from_seq_region(ln(h1))                       # f'
out = h1 + reduce_scatter_to_seq_region(mlp(m, W1, W2))           # g'
comm_report("forward")
ok_fwd = same(out, Y_ref[sl])

out.backward(gY[sl])
comm_report("backward")
ok_dx = same(X_loc.grad, Xr.grad[sl])

status = "OK" if (ok_fwd and ok_dx) else "FAIL"
print(f"rank {rank}: forward 分片对拍 {ok_fwd}，dX 分片对拍 {ok_dx} [{status}]",
      flush=True)
dist.destroy_process_group()
