# exp16_ep_moe.py —— 实验一：手写 EP 版 MoE——dispatch/combine 两次 all_to_all，与单卡对比
# 4 个专家、每卡 2 个；每个 token 经 router 选 top-2 专家。
# 前向：按目的地打包 → 先互换计数（各 rank 会收到多少行）→ all_to_all 寄 token
#       → 各卡只算自己的专家 → all_to_all 寄回 → 按门控分加权相加。
# 反向：梯度沿原路返回，又是两次 all_to_all（自定义 autograd.Function 里实现）。
# 对比：本卡 out / X.grad 与单卡参考一致；本卡专家的梯度与"两卡数据合算"的参考一致。
# GPU 机器上默认 cuda/nccl，其余机器自动退回 cpu/gloo（也可用 --device cpu 强制）。
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import lab_init

dev, rank, world = lab_init("实验一：手写 EP 版 MoE")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

# ---- 通信计数器：按 op 记次数与发送侧字节数，每个 rank 自己打印、打印后清零 ----
_CNT = {}
def _rec(op, t):
    _CNT.setdefault(op, [0, 0])
    _CNT[op][0] += 1
    _CNT[op][1] += t.numel() * t.element_size()
def comm_reset():
    _CNT.clear()
def comm_report(tag):
    def fmt(b):
        return f"{b / 2**10:.1f} KiB" if b >= 1024 else f"{b} B"
    parts = "  ".join(f"{k}×{v[0]}（{fmt(v[1])}）" for k, v in sorted(_CNT.items()))
    print(f"[通信统计] rank{rank} {tag}：{parts}", flush=True)
    _CNT.clear()

torch.manual_seed(42)                  # 两卡同种子 → 同一份 router 权重与专家全量权重
H, FF, E, T, K = 256, 512, 4, 64, 2    # hidden、专家 FFN 宽度、专家数、每卡 token 数、top-k
EPR = E // world                       # 每卡 2 个专家
Wr_full = torch.randn(H, E, device=dev) / H**0.5   # router 权重：小矩阵，两卡各复制一份
W1_full = [torch.randn(H, FF, device=dev) / H**0.5 for _ in range(E)]
W2_full = [torch.randn(FF, H, device=dev) / FF**0.5 for _ in range(E)]

torch.manual_seed(1000 + rank)         # 两卡不同数据：EP 组里每张卡处理自己的 token
X = torch.randn(T, H, device=dev, requires_grad=True)
Yt = torch.randn(T, H, device=dev)
Wr = Wr_full.clone().requires_grad_(True)
W1 = [W1_full[rank * EPR + i].clone().requires_grad_(True) for i in range(EPR)]
W2 = [W2_full[rank * EPR + i].clone().requires_grad_(True) for i in range(EPR)]


class _AllToAll(torch.autograd.Function):
    """按 send_splits 把行寄给各 rank、按 recv_splits 收齐；
    反向把梯度按相反方向寄回（splits 对调）——寄件的反向还是寄件。"""

    @staticmethod
    def forward(ctx, x, send_splits, recv_splits):
        ctx.send_splits, ctx.recv_splits = send_splits, recv_splits
        out = torch.empty(sum(recv_splits), *x.shape[1:], dtype=x.dtype, device=x.device)
        _rec("all_to_all", x)
        dist.all_to_all_single(out, x.contiguous(), recv_splits, send_splits)
        return out

    @staticmethod
    def backward(ctx, gout):
        gin = torch.empty(sum(ctx.send_splits), *gout.shape[1:], dtype=gout.dtype,
                          device=gout.device)
        _rec("all_to_all", gout)
        dist.all_to_all_single(gin, gout.contiguous(), ctx.send_splits, ctx.recv_splits)
        return gin, None, None


def a2a_meta(x, send_splits, recv_splits):
    """ids/counts 等整数张量的 all_to_all：不参与反向，不需要 autograd。"""
    out = torch.empty(sum(recv_splits), dtype=torch.int64, device=x.device)
    _rec("a2a(ids)", x)
    dist.all_to_all_single(out, x, recv_splits, send_splits)
    return out


# ---------- 前向 ----------
scores = F.softmax(X @ Wr, dim=-1)         # [T, E] router 打分：每卡只给自己的 token 打分
val, idx = scores.topk(K, dim=-1)          # 每个 token 选 K 个专家，附带门控分
tok = torch.arange(T, device=dev).repeat_interleave(K)  # 展开成 T*K 份投递单
exp_id = idx.reshape(-1)
gate = val.reshape(-1)
dest = exp_id // EPR                       # 每份投递单寄往哪张卡

counts = torch.bincount(dest, minlength=world)      # 我寄给每张卡多少行
peer = torch.empty(world, dtype=torch.int64, device=dev)
_rec("a2a(counts)", counts)
dist.all_to_all_single(peer, counts)                # 每张卡寄给我多少行——先问一声再寄
send_splits = counts.tolist()
recv_splits = peer.tolist()

order = torch.argsort(dest, stable=True)            # 按目的地归拢，准备打包
send_buf = X[tok[order]]                            # 打包后的 token 向量
ids_buf = exp_id[order]                             # 对应的专家编号（接收方分拣要用）

recv_buf = _AllToAll.apply(send_buf, send_splits, recv_splits)  # dispatch
ids_recv = a2a_meta(ids_buf, send_splits, recv_splits)

res_buf = torch.zeros_like(recv_buf)                # 各卡只算自己家里的专家
for i in range(EPR):
    sel = (ids_recv == rank * EPR + i).nonzero().squeeze(-1)
    xe = recv_buf.index_select(0, sel)
    ye = F.gelu(xe @ W1[i]) @ W2[i]
    res_buf = res_buf.index_copy(0, sel, ye)

returned = _AllToAll.apply(res_buf, recv_splits, send_splits)   # combine：按原路寄回
got = torch.zeros(T * K, H, device=dev).index_copy(0, order, returned)  # 拆包：放回投递单顺序
out = torch.zeros(T, H, device=dev).index_add(0, tok, got * gate[:, None])  # 按门控分加权相加
loss = ((out - Yt) ** 2).mean()

per_expert = torch.bincount(ids_recv, minlength=E)[rank * EPR:(rank + 1) * EPR]
print(f"rank{rank} 本卡专家各收到 {per_expert.tolist()} 个 token"
      f"（全组共寄出 {world * T * K} 份）", flush=True)
comm_report("forward（dispatch + combine + 计数/编号）")

# ---------- 反向 ----------
comm_reset()
loss.backward()
comm_report("backward")

_rec("all_reduce", Wr.grad)
dist.all_reduce(Wr.grad)                 # router 是复制参数，梯度要在 EP 组内求和
print(f"rank{rank} router 梯度 all_reduce：{Wr.grad.numel() * 4 / 2**10:.1f} KiB", flush=True)

# ---------- 参考：单卡 MoE。out/X.grad 用本卡数据；专家梯度、router 梯度要算上两卡数据 ----------
def moe_ref(Xr, Wrr, W1r, W2r):
    scores = F.softmax(Xr @ Wrr, dim=-1)
    val, idx = scores.topk(K, dim=-1)
    tok = torch.arange(T, device=Xr.device).repeat_interleave(K)
    eid = idx.reshape(-1)
    w = val.reshape(-1)
    o = torch.zeros(T, H, device=Xr.device)
    for e in range(E):
        sel = (eid == e).nonzero().squeeze(-1)
        ye = F.gelu(Xr[tok[sel]] @ W1r[e]) @ W2r[e]
        o = o.index_add(0, tok[sel], ye * w[sel, None])
    return o

gWr = torch.zeros_like(Wr_full)
gW1 = [torch.zeros_like(w) for w in W1_full]
gW2 = [torch.zeros_like(w) for w in W2_full]
out_ref = Xgrad_ref = None
for j in range(world):                   # 复现第 j 张卡的数据，把两卡的梯度都累加进参考
    torch.manual_seed(1000 + j)
    Xj = torch.randn(T, H, device=dev).requires_grad_(True)
    Yj = torch.randn(T, H, device=dev)
    Wrr = Wr_full.clone().requires_grad_(True)
    W1r = [w.clone().requires_grad_(True) for w in W1_full]
    W2r = [w.clone().requires_grad_(True) for w in W2_full]
    lj = ((moe_ref(Xj, Wrr, W1r, W2r) - Yj) ** 2).mean()
    lj.backward()
    gWr += Wrr.grad
    for e in range(E):
        gW1[e] += W1r[e].grad
        gW2[e] += W2r[e].grad
    if j == rank:                        # 本卡数据的前向输出与 X 梯度参考
        out_ref = moe_ref(Xj.detach(), Wr_full, W1_full, W2_full)
        Xgrad_ref = Xj.grad

def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5)

ok_out = same(out.detach(), out_ref)
ok_x = same(X.grad, Xgrad_ref)
ok_w = all(same(W1[i].grad, gW1[rank * EPR + i]) and same(W2[i].grad, gW2[rank * EPR + i])
           for i in range(EPR))
ok_wr = same(Wr.grad, gWr)
status = "OK" if all([ok_out, ok_x, ok_w, ok_wr]) else "FAIL"
print(f"rank{rank}：输出对比 {ok_out}，X 梯度对比 {ok_x}，"
      f"本卡专家梯度对比 {ok_w}，router 梯度对比 {ok_wr} [{status}]", flush=True)

dist.destroy_process_group()
