# exp8_layer.py —— 实验二：完整一层 Transformer，亲手数出 4 次 all_reduce
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import (lab_init, copy_to_tp_region, reduce_from_tp_region,
                     comm_reset, comm_report)

dev, rank, world = lab_init("实验二：一层 Transformer 的 4 次通信")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

torch.manual_seed(0)
S, B, H, NH = 256, 2, 2048, 8           # 序列长、batch、hidden、注意力头数
HD = H // NH
X = torch.randn(S, B, H, device=dev)
Att_f = [torch.randn(H, H, device=dev) / H**0.5 for _ in range(4)]  # Wq Wk Wv Wo
W1f = torch.randn(H, 4 * H, device=dev) / H**0.5
W2f = torch.randn(4 * H, H, device=dev) / (4 * H)**0.5

def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-4)

def attn(x, wq, wk, wv, wo):
    q, k, v = x @ wq, x @ wk, x @ wv     # TP 时 wq/wk/wv 已按列切 → 本卡一半的头
    nh = q.size(-1) // HD                # 头数从张量宽度推：同一份代码两种身份
    q = q.view(S, B, nh, HD).permute(1, 2, 0, 3)     # [B, nh, S, HD]
    k = k.view(S, B, nh, HD).permute(1, 2, 0, 3)
    v = v.view(S, B, nh, HD).permute(1, 2, 0, 3)
    o = F.scaled_dot_product_attention(q, k, v)      # 头间独立，本地算
    return o.permute(2, 0, 1, 3).reshape(S, B, nh * HD) @ wo

def mlp(x, w1, w2):
    return F.gelu(x @ w1) @ w2

def layer(x, att, w1, w2, tp):
    if tp:
        a = copy_to_tp_region(x)                          # f1：前向恒等
        h_attn = reduce_from_tp_region(attn(a, *att))     # g1：all-reduce ①
    else:
        h_attn = attn(x, *att)
    h1 = x + h_attn                                       # 残差相加：逐元素，免费
    if tp:
        m = copy_to_tp_region(h1)                         # f2：前向恒等
        h_mlp = reduce_from_tp_region(mlp(m, w1, w2))     # g2：all-reduce ②
    else:
        h_mlp = mlp(h1, w1, w2)
    return h1 + h_mlp

# ---- 参考：完整权重，等价单卡 ----
Xr = X.clone().requires_grad_(True)
Attr = [w.clone().requires_grad_(True) for w in Att_f]
W1r, W2r = W1f.clone().requires_grad_(True), W2f.clone().requires_grad_(True)
Y_ref = layer(Xr, Attr, W1r, W2r, tp=False)
gY = torch.randn(S, B, H, device=dev)
Y_ref.backward(gY)

# ---- TP：Wq/Wk/Wv/W1 按列切，Wo/W2 按行切 ----
sub, half = H // world, (4 * H) // world
Wq, Wk, Wv = [Att_f[i][:, rank*sub:(rank+1)*sub].clone().requires_grad_(True)
              for i in range(3)]
Wo = Att_f[3][rank*sub:(rank+1)*sub, :].clone().requires_grad_(True)
W1 = W1f[:, rank*half:(rank+1)*half].clone().requires_grad_(True)
W2 = W2f[rank*half:(rank+1)*half, :].clone().requires_grad_(True)

X_tp = X.clone().requires_grad_(True)
comm_reset()
Y_tp = layer(X_tp, [Wq, Wk, Wv, Wo], W1, W2, tp=True)
comm_report("forward")
ok_fwd = same(Y_tp, Y_ref)

Y_tp.backward(gY)
comm_report("backward")
ok_dx = same(X_tp.grad, Xr.grad)
ok_wo = same(Wo.grad, Attr[3].grad[rank*sub:(rank+1)*sub, :])
ok_w1 = same(W1.grad, W1r.grad[:, rank*half:(rank+1)*half])

status = "OK" if all([ok_fwd, ok_dx, ok_wo, ok_w1]) else "FAIL"
print(f"rank {rank}: forward 对拍 {ok_fwd}，dX 对拍 {ok_dx}，"
      f"dWo/dW1 分片对拍 {ok_wo and ok_w1} [{status}]", flush=True)
dist.destroy_process_group()
