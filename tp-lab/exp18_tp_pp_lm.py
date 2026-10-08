# exp18_tp_pp_lm.py —— 实验一：TP×PP 组合——完整小 LM 切上 2×2，与单卡对拍
# 一个完整的迷你语言模型：词嵌入 → 2 层 transformer → 输出头（与词嵌入共享权重）→ CE loss。
# 4 个进程：rank = tp + 2·pp —— TP 组是同段两张卡（列切/行切），PP 是两段接力。
# 前向：词嵌入各卡查半张词表再 all-reduce → 每层 f 进 g 出 → 边界激活 send 给下一段
#       → 末段输出头列切算 logits、all-gather 拼全词表 → CE loss。
# 反向：梯度沿原路 recv 回来；共享的 embedding 权重梯度在"首段↔末段"组里 all-reduce。
# 对拍：loss、边界激活、边界梯度、各层权重梯度分片、embedding 梯度分片，全部对单卡参考。
# GPU 机器上默认 cuda/nccl，其余机器自动退回 cpu/gloo（也可用 --device cpu 强制）。
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import (lab_init, _rec, comm_reset, comm_report_rank,
                     copy_to_region, reduce_from_region, gather_from_region)

dev, rank, world = lab_init("实验一：TP×PP 组合的完整小 LM")
assert world == 4, "请用 torchrun --nproc_per_node=4 运行"

# ---- 坐标与进程组：rank = tp + 2·pp（TP 最内、PP 最外）----
tp, pp = rank % 2, rank // 2
# 修补：所有进程按同一顺序建全部组，各卡只留自己的句柄（§2.1 枚举写法 / 附录 A 第一条）
tp_group = pp_group = emb_group = None
for k in range(2):                                # TP 组：{0,1}、{2,3}
    m = [2 * k, 2 * k + 1]
    g = dist.new_group(m)
    if rank in m:
        tp_group = g
for i in range(2):                                # PP 车道：{0,2}、{1,3}
    m = [i, i + 2]
    g = dist.new_group(m)
    if rank in m:
        pp_group = g
for i in range(2):                                # embedding 组：{0,2}、{1,3}
    m = [i, i + 2]
    g = dist.new_group(m)
    if rank in m:
        emb_group = g
print(f"rank{rank}：坐标 (tp{tp}, pp{pp})｜TP 组 {[2 * pp, 2 * pp + 1]}"
      f"｜PP 车道 {[tp, tp + 2]}｜embedding 组 {[tp, tp + 2]}", flush=True)

V, H, NH, FF, T = 64, 256, 4, 1024, 64    # 词表、hidden、头数、FFN 宽度、token 数
HD = H // NH

torch.manual_seed(42)                     # 四卡同种子 → 同一份完整权重
E_full = torch.randn(V, H, device=dev) / H**0.5
LAY_full = [{k: torch.randn(H, H, device=dev) / H**0.5 for k in ("Wq", "Wk", "Wv", "Wo")}
            | {"W1": torch.randn(H, FF, device=dev) / H**0.5,
               "W2": torch.randn(FF, H, device=dev) / FF**0.5} for _ in range(2)]
torch.manual_seed(1000)                   # dp=1：数据处处相同（PP 切的是模型不是数据）
tok = torch.randint(0, V, (T,), device=dev)
tgt = torch.randint(0, V, (T,), device=dev)

# ---- 各卡切片：E 沿词表行切；Wq/Wk/Wv/W1 列切；Wo/W2 行切；层号 = 段号 ----
sv, sh, sf = V // 2, H // 2, FF // 2
E = E_full[tp * sv:(tp + 1) * sv].clone().requires_grad_(True)
Wf = LAY_full[pp]
W = dict(Wq=Wf["Wq"][:, tp * sh:(tp + 1) * sh].clone().requires_grad_(True),
         Wk=Wf["Wk"][:, tp * sh:(tp + 1) * sh].clone().requires_grad_(True),
         Wv=Wf["Wv"][:, tp * sh:(tp + 1) * sh].clone().requires_grad_(True),
         Wo=Wf["Wo"][tp * sh:(tp + 1) * sh, :].clone().requires_grad_(True),
         W1=Wf["W1"][:, tp * sf:(tp + 1) * sf].clone().requires_grad_(True),
         W2=Wf["W2"][tp * sf:(tp + 1) * sf, :].clone().requires_grad_(True))
full_mb = (2 * E_full.numel() + 2 * sum(w.numel() for w in LAY_full[0].values())) * 4 / 2**20
mine_mb = (E.numel() + sum(w.numel() for w in W.values())) * 4 / 2**20
print(f"rank{rank}：本卡权重 {mine_mb:.2f} MiB（全模型 {full_mb:.2f} MiB"
      f"（embedding 首末段各存一份），÷tp÷pp = ÷4）", flush=True)


def ln(x):
    return F.layer_norm(x, (x.size(-1),))          # 无参数 LN：逐 token，免费切分


def attn(x, Wq, Wk, Wv, Wo):
    nh = Wq.size(1) // HD                          # 从宽度推本卡头数：同一函数两种身份
    q = (x @ Wq).view(T, nh, HD).transpose(0, 1)
    k = (x @ Wk).view(T, nh, HD).transpose(0, 1)
    v = (x @ Wv).view(T, nh, HD).transpose(0, 1)
    o = F.scaled_dot_product_attention(q, k, v)    # 头间独立，本地算
    return o.transpose(0, 1).reshape(T, nh * HD) @ Wo


def mlp(x, W1, W2):
    return F.gelu(x @ W1) @ W2


def layer_ref(x, Wf):                              # 单卡参考用：不切分
    h = x + attn(ln(x), Wf["Wq"], Wf["Wk"], Wf["Wv"], Wf["Wo"])
    return h + mlp(ln(h), Wf["W1"], Wf["W2"])


def tp_layer(x, W):                                # TP 版：入口复制、出口 all-reduce，通信在 TP 组内
    h = x + reduce_from_region(attn(copy_to_region(ln(x), tp_group, "tp"),
                                    W["Wq"], W["Wk"], W["Wv"], W["Wo"]), tp_group, "tp")
    return h + reduce_from_region(mlp(copy_to_region(ln(h), tp_group, "tp"),
                                      W["W1"], W["W2"]), tp_group, "tp")


def vocab_embed(tok, E_loc):                       # 词表并行嵌入：各卡只查自己半张词表
    lo = tp * sv
    mask = (tok < lo) | (tok >= lo + sv)
    idx = (tok - lo).clamp(0, sv - 1)              # 不在本卡的先夹到合法范围，再清零
    return F.embedding(idx, E_loc).masked_fill(mask[:, None], 0.0)


# ---------- 前向 + 反向 ----------
comm_reset()
if pp == 0:                                        # 首段：嵌入 + 第 0 层
    emb = reduce_from_region(vocab_embed(tok, E), tp_group, "tp")   # 半词表求和
    h = tp_layer(emb, W)
    act = h.detach().contiguous()                  # 边界激活：发给下一段
    _rec("send[pp]", act)
    dist.send(act, group=pp_group, group_dst=1)
else:                                              # 末段：收激活 + 第 1 层 + 输出头
    act = torch.empty(T, H, device=dev)
    _rec("recv[pp]", act)
    dist.recv(act, group=pp_group, group_src=0)
    act.requires_grad_(True)
    h = tp_layer(act, W)
    h_in = copy_to_region(h, tp_group, "tp")       # 输出头 = 列切层：入口原样复制（反向把两半梯度求和）
    logits = gather_from_region(h_in @ E.T, tp_group, "tp")   # 词表列切，出口拼全
    loss = F.cross_entropy(logits, tgt)
comm_report_rank("forward")

if pp == 1:
    loss.backward()
    dA = act.grad.contiguous()                     # 边界梯度：发回上一段
    _rec("send[pp]", dA)
    dist.send(dA, group=pp_group, group_dst=0)
else:
    dA = torch.empty(T, H, device=dev)
    _rec("recv[pp]", dA)
    dist.recv(dA, group=pp_group, group_src=1)
    h.backward(dA)
comm_report_rank("backward")

# embedding 权重两段共享：嵌入侧梯度与输出头侧梯度在 embedding 组内求和
_rec("all_reduce[emb]", E.grad)
dist.all_reduce(E.grad, group=emb_group)
comm_report_rank("embedding 组")

# ---------- 单卡参考（每卡本地各算一遍）----------
Er = E_full.clone().requires_grad_(True)
Lr = [{k: v.clone().requires_grad_(True) for k, v in wf.items()} for wf in LAY_full]
emb_r = F.embedding(tok, Er)
h0r = layer_ref(emb_r, Lr[0])
h0r.retain_grad()                                  # 它就是"边界激活"的参考
logits_r = layer_ref(h0r, Lr[1]) @ Er.T            # 输出头与嵌入共享 Er
loss_r = F.cross_entropy(logits_r, tgt)
loss_r.backward()

def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5)

ok_e = same(E.grad, Er.grad[tp * sv:(tp + 1) * sv])
ok_w = (same(W["Wq"].grad, Lr[pp]["Wq"].grad[:, tp * sh:(tp + 1) * sh]) and
        same(W["Wo"].grad, Lr[pp]["Wo"].grad[tp * sh:(tp + 1) * sh, :]) and
        same(W["W1"].grad, Lr[pp]["W1"].grad[:, tp * sf:(tp + 1) * sf]) and
        same(W["W2"].grad, Lr[pp]["W2"].grad[tp * sf:(tp + 1) * sf, :]))
if pp == 0:
    ok_act, ok_da = same(act, h0r.detach()), same(dA, h0r.grad)
    ok = all([ok_act, ok_da, ok_w, ok_e])
    print(f"rank{rank}：边界激活对比 {ok_act}，边界梯度对比 {ok_da}，"
          f"本段层权重梯度对比 {ok_w}，embedding 梯度对比 {ok_e} [{'OK' if ok else 'FAIL'}]",
          flush=True)
else:
    ok_loss = same(loss.detach(), loss_r.detach())
    ok = all([ok_loss, ok_w, ok_e])
    print(f"rank{rank}：loss 对比 {ok_loss}，"
          f"本段层权重梯度对比 {ok_w}，embedding 梯度对比 {ok_e} [{'OK' if ok else 'FAIL'}]",
          flush=True)

dist.destroy_process_group()
