# exp19_3d_lm.py —— 实验二：TP×DP×PP 完整组合——2×2×2 共 8 个进程，与"两路数据合算"对拍
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import (lab_init, _rec, comm_reset, comm_report_rank,
                     copy_to_region, reduce_from_region, gather_from_region)

dev, rank, world = lab_init("实验二：TP×DP×PP 完整组合")
assert world == 8, "请用 torchrun --nproc_per_node=8 运行"

# ---- 三维坐标与四个进程组（所有进程都要调用全部 new_group）----
tp, dp, pp = rank % 2, (rank // 2) % 2, rank // 4
# 修补：所有进程按同一顺序建全部组，各卡只留自己的句柄（§2.1 枚举写法 / 附录 A 第一条）
tp_group = dp_group = pp_group = emb_group = None
for k in range(2):                       # 4 个 TP 组
    for j in range(2):
        m = [4 * k + 2 * j, 4 * k + 2 * j + 1]
        g = dist.new_group(m)
        if rank in m:
            tp_group = g
for k in range(2):                       # 4 个 DP 组
    for i in range(2):
        m = [4 * k + i, 4 * k + i + 2]
        g = dist.new_group(m)
        if rank in m:
            dp_group = g
for j in range(2):                       # 4 条 PP 车道
    for i in range(2):
        m = [2 * j + i, 2 * j + i + 4]
        g = dist.new_group(m)
        if rank in m:
            pp_group = g
for j in range(2):                       # 4 个 embedding 组
    for i in range(2):
        m = [2 * j + i, 2 * j + i + 4]
        g = dist.new_group(m)
        if rank in m:
            emb_group = g
print(f"rank{rank}：坐标 (tp{tp}, dp{dp}, pp{pp})｜TP {[4 * pp + 2 * dp, 4 * pp + 2 * dp + 1]}"
      f"｜DP {[4 * pp + tp, 4 * pp + tp + 2]}｜PP 车道 {[2 * dp + tp, 2 * dp + tp + 4]}",
      flush=True)

V, H, NH, FF, T = 64, 256, 4, 1024, 64
HD = H // NH

torch.manual_seed(42)                     # 八卡同种子 → 同一份完整权重
E_full = torch.randn(V, H, device=dev) / H**0.5
LAY_full = [{k: torch.randn(H, H, device=dev) / H**0.5 for k in ("Wq", "Wk", "Wv", "Wo")}
            | {"W1": torch.randn(H, FF, device=dev) / H**0.5,
               "W2": torch.randn(FF, H, device=dev) / FF**0.5} for _ in range(2)]
torch.manual_seed(1000 + dp)              # DP 的本义：两条车道吃不同的数据
tok = torch.randint(0, V, (T,), device=dev)
tgt = torch.randint(0, V, (T,), device=dev)

# ---- 各卡切片：与 exp18 相同 ----
sv, sh, sf = V // 2, H // 2, FF // 2
E = E_full[tp * sv:(tp + 1) * sv].clone().requires_grad_(True)
Wf = LAY_full[pp]
W = dict(Wq=Wf["Wq"][:, tp * sh:(tp + 1) * sh].clone().requires_grad_(True),
         Wk=Wf["Wk"][:, tp * sh:(tp + 1) * sh].clone().requires_grad_(True),
         Wv=Wf["Wv"][:, tp * sh:(tp + 1) * sh].clone().requires_grad_(True),
         Wo=Wf["Wo"][tp * sh:(tp + 1) * sh, :].clone().requires_grad_(True),
         W1=Wf["W1"][:, tp * sf:(tp + 1) * sf].clone().requires_grad_(True),
         W2=Wf["W2"][tp * sf:(tp + 1) * sf, :].clone().requires_grad_(True))
mine_mb = (E.numel() + sum(w.numel() for w in W.values())) * 4 / 2**20
full_mb = (2 * E_full.numel() + 2 * sum(w.numel() for w in LAY_full[0].values())) * 4 / 2**20
print(f"rank{rank}：本卡权重 {mine_mb:.2f} MiB（全模型 {full_mb:.2f} MiB"
      f"（embedding 首末段各存一份）；÷tp÷pp = ÷4，DP 维度上是复制）", flush=True)


def ln(x):
    return F.layer_norm(x, (x.size(-1),))


def attn(x, Wq, Wk, Wv, Wo):
    nh = Wq.size(1) // HD
    q = (x @ Wq).view(T, nh, HD).transpose(0, 1)
    k = (x @ Wk).view(T, nh, HD).transpose(0, 1)
    v = (x @ Wv).view(T, nh, HD).transpose(0, 1)
    o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(0, 1).reshape(T, nh * HD) @ Wo


def mlp(x, W1, W2):
    return F.gelu(x @ W1) @ W2


def layer_ref(x, Wf):
    h = x + attn(ln(x), Wf["Wq"], Wf["Wk"], Wf["Wv"], Wf["Wo"])
    return h + mlp(ln(h), Wf["W1"], Wf["W2"])


def tp_layer(x, W):
    h = x + reduce_from_region(attn(copy_to_region(ln(x), tp_group, "tp"),
                                    W["Wq"], W["Wk"], W["Wv"], W["Wo"]), tp_group, "tp")
    return h + reduce_from_region(mlp(copy_to_region(ln(h), tp_group, "tp"),
                                      W["W1"], W["W2"]), tp_group, "tp")


def vocab_embed(tok, E_loc):
    lo = tp * sv
    mask = (tok < lo) | (tok >= lo + sv)
    idx = (tok - lo).clamp(0, sv - 1)
    return F.embedding(idx, E_loc).masked_fill(mask[:, None], 0.0)


# ---------- 前向 + 反向 ----------
comm_reset()
if pp == 0:
    emb = reduce_from_region(vocab_embed(tok, E), tp_group, "tp")
    h = tp_layer(emb, W)
    act = h.detach().contiguous()
    _rec("send[pp]", act)
    dist.send(act, group=pp_group, group_dst=1)
else:
    act = torch.empty(T, H, device=dev)
    _rec("recv[pp]", act)
    dist.recv(act, group=pp_group, group_src=0)
    act.requires_grad_(True)
    h = tp_layer(act, W)
    h_in = copy_to_region(h, tp_group, "tp")
    logits = gather_from_region(h_in @ E.T, tp_group, "tp")
    loss = F.cross_entropy(logits, tgt)
comm_report_rank("forward")

if pp == 1:
    loss.backward()
    dA = act.grad.contiguous()
    _rec("send[pp]", dA)
    dist.send(dA, group=pp_group, group_dst=0)
else:
    dA = torch.empty(T, H, device=dev)
    _rec("recv[pp]", dA)
    dist.recv(dA, group=pp_group, group_src=1)
    h.backward(dA)
comm_report_rank("backward")

# ① 共享的 embedding 权重：先在 embedding 组内求和
_rec("all_reduce[emb]", E.grad)
dist.all_reduce(E.grad, group=emb_group)
comm_report_rank("embedding 组")

# ② DP 组：每块权重分片的梯度跨两条数据通道求和
params = [E, W["Wq"], W["Wk"], W["Wv"], W["Wo"], W["W1"], W["W2"]]
for p in params:
    _rec("all_reduce[dp]", p.grad)
    dist.all_reduce(p.grad, group=dp_group)
comm_report_rank("DP 组")

# ---------- 参考：单卡把两个 lane 各算一遍，梯度相加 ----------
gE = torch.zeros_like(E_full)
gL = [{k: torch.zeros_like(v) for k, v in wf.items()} for wf in LAY_full]
loss_ref = None
for j in range(2):
    torch.manual_seed(1000 + j)           # 复现 lane j 的数据
    tokj = torch.randint(0, V, (T,), device=dev)
    tgtj = torch.randint(0, V, (T,), device=dev)
    Ej = E_full.clone().requires_grad_(True)
    Lj = [{k: v.clone().requires_grad_(True) for k, v in wf.items()} for wf in LAY_full]
    hj = layer_ref(layer_ref(F.embedding(tokj, Ej), Lj[0]), Lj[1])
    lossj = F.cross_entropy(hj @ Ej.T, tgtj)
    lossj.backward()
    gE += Ej.grad
    for i in range(2):
        for k in gL[i]:
            gL[i][k] += Lj[i][k].grad
    if j == dp:
        loss_ref = lossj.detach()

def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5)

ok_e = same(E.grad, gE[tp * sv:(tp + 1) * sv])
ok_w = (same(W["Wq"].grad, gL[pp]["Wq"][:, tp * sh:(tp + 1) * sh]) and
        same(W["Wo"].grad, gL[pp]["Wo"][tp * sh:(tp + 1) * sh, :]) and
        same(W["W1"].grad, gL[pp]["W1"][:, tp * sf:(tp + 1) * sf]) and
        same(W["W2"].grad, gL[pp]["W2"][tp * sf:(tp + 1) * sf, :]))
if pp == 1:
    ok_loss = same(loss.detach(), loss_ref)
    ok = all([ok_loss, ok_w, ok_e])
    print(f"rank{rank}：loss 对比 {ok_loss}，层权重梯度对比 {ok_w}，"
          f"embedding 梯度对比 {ok_e} [{'OK' if ok else 'FAIL'}]", flush=True)
else:
    ok = all([ok_w, ok_e])
    print(f"rank{rank}：层权重梯度对比 {ok_w}，"
          f"embedding 梯度对比 {ok_e} [{'OK' if ok else 'FAIL'}]", flush=True)

dist.destroy_process_group()
