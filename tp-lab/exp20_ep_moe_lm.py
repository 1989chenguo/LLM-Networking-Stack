# exp20_ep_moe_lm.py —— 实验三：DP×PP 骨架上叠加 EP——MoE 小 LM 切上 4×2，专家再切 EP2，与单卡对拍
# 一个 MoE 版的完整小 LM：词嵌入 → 2 层（attention + 4 专家 top-1 MoE）→ 共享输出头 → CE loss。
# 8 个进程：rank = dp + 4·pp —— DP4 四条车道、PP2 两段；EP 组从 DP 组里划（e=2、edp=2）。
# 前向：MoE 层内 all_to_all 分发 token、本地专家计算、all_to_all 收回；段间 send/recv。
# 反向：all_to_all 的反向仍是 all_to_all（autograd 自动串起）；step 末三次求和：
#   ① embedding 组（首段↔末段、同车道）② DP 组（非专家参数 + router）③ EDP 组（专家权重）。
# 对拍：loss、边界激活/梯度、attention 与 router 梯度、专家梯度、embedding 梯度，全部对单卡参考。
# GPU 机器上默认 cuda/nccl（需要 8 卡），其余机器自动退回 cpu/gloo（--device cpu 可强制）。
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import lab_init, _rec, comm_reset, comm_report_rank

dev, rank, world = lab_init("实验三：DP×PP 上的 EP——MoE 小 LM")
assert world == 8, "请用 torchrun --nproc_per_node=8 运行"

# ---- 坐标与进程组：rank = dp + 4·pp；dp 再切成 ep + 2·edp（EP 在内、EDP 在外）----
dp, pp = rank % 4, rank // 4
ep, edp = dp % 2, dp // 2
dp_group = ep_group = edp_group = pp_group = emb_group = None
for k in range(2):                              # 每一段内：1 个 DP 组、2 个 EP 组、2 个 EDP 组
    for m in [[4 * k + j for j in range(4)]]:   # DP 组：同段四条车道（稠密参数梯度在这里求和）
        grp = dist.new_group(m)
        dp_group = grp if rank in m else dp_group
    for m in [[4 * k + 2 * j, 4 * k + 2 * j + 1] for j in range(2)]:   # EP 组：all_to_all 在这里做
        grp = dist.new_group(m)
        ep_group = grp if rank in m else ep_group
    for m in [[4 * k + i, 4 * k + i + 2] for i in range(2)]:           # EDP 组：专家梯度在这里求和
        grp = dist.new_group(m)
        edp_group = grp if rank in m else edp_group
for j in range(4):                              # 四条 PP 车道；embedding 组与车道同成员（pp=2 的特例）
    grp = dist.new_group([j, j + 4])
    pp_group = grp if rank in [j, j + 4] else pp_group
    grp = dist.new_group([j, j + 4])
    emb_group = grp if rank in [j, j + 4] else emb_group
print(f"rank{rank}：坐标 (dp{dp}, pp{pp})，EP 内编号 ep{ep}｜DP 组 {[4 * pp + j for j in range(4)]}"
      f"｜EP 组 {[4 * pp + 2 * edp, 4 * pp + 2 * edp + 1]}｜EDP 组 {[4 * pp + ep, 4 * pp + ep + 2]}"
      f"｜PP 车道/emb 组 {[dp, dp + 4]}", flush=True)

V, H, NH, FF, T, NE = 64, 256, 4, 1024, 64, 4
EL = NE // 2                                    # 每卡持有的专家数
HD = H // NH

torch.manual_seed(42)                           # 八卡同种子 → 同一份完整权重
E_full = torch.randn(V, H, device=dev) / H**0.5
LAY_full = [{k: torch.randn(H, H, device=dev) / H**0.5 for k in ("Wq", "Wk", "Wv", "Wo")}
            | {"Wr": torch.randn(H, NE, device=dev) / H**0.5,
               "EX": [(torch.randn(H, FF, device=dev) / H**0.5,
                       torch.randn(FF, H, device=dev) / FF**0.5) for _ in range(NE)]}
            for _ in range(2)]
torch.manual_seed(1000 + dp)                    # DP 的本义：四条车道吃不同的数据
tok = torch.randint(0, V, (T,), device=dev)
tgt = torch.randint(0, V, (T,), device=dev)

# ---- 各卡切片：E 整份（首段嵌入、末段输出头）；attention/router 整份（DP 复制）；
# ---- 专家只拿自己那 EL 个（EP 切分）；层号 = 段号（PP 切分）----
E = E_full.clone().requires_grad_(True)
Wf = LAY_full[pp]
W = {k: Wf[k].clone().requires_grad_(True) for k in ("Wq", "Wk", "Wv", "Wo", "Wr")}
EX = [(w1.clone().requires_grad_(True), w2.clone().requires_grad_(True))
      for (w1, w2) in Wf["EX"][ep * EL:(ep + 1) * EL]]
mine_mb = (E.numel() + sum(w.numel() for w in W.values())
           + sum(w.numel() for pair in EX for w in pair)) * 4 / 2**20
full_mb = (2 * E_full.numel()
           + 2 * (sum(Wf[k].numel() for k in ("Wq", "Wk", "Wv", "Wo", "Wr"))
                  + sum(w.numel() for pair in Wf["EX"] for w in pair))) * 4 / 2**20
print(f"rank{rank}：本卡权重 {mine_mb:.2f} MiB（全模型 {full_mb:.2f} MiB（embedding 首末段各存一份）；"
      f"专家 ÷2（EP），其余在 DP 维复制）", flush=True)


def ln(x):
    return F.layer_norm(x, (x.size(-1),))


def attn(x, Wq, Wk, Wv, Wo):
    nh = Wq.size(1) // HD
    q = (x @ Wq).view(T, nh, HD).transpose(0, 1)
    k = (x @ Wk).view(T, nh, HD).transpose(0, 1)
    v = (x @ Wv).view(T, nh, HD).transpose(0, 1)
    o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(0, 1).reshape(T, nh * HD) @ Wo


EP_MEM = [4 * pp + 2 * edp, 4 * pp + 2 * edp + 1]   # 本卡 EP 组的全局成员


def _a2a(x, s_send, s_recv):                    # all_to_all 本体：组内按 split 互换行块
    squeeze = x.dim() == 1                      # 一维输入（条数交换）补成二维再发
    if squeeze:
        x = x[:, None]
    xs = list(x.split(s_send))
    ys = [torch.empty(int(n), x.size(1), dtype=x.dtype, device=dev) for n in s_recv]
    ops = []
    for i, peer in enumerate(EP_MEM):
        if peer == rank:
            ys[i].copy_(xs[i])                  # 发给自己的那一块直接拷贝
            continue
        if s_send[i] > 0:
            ops.append(dist.P2POp(dist.isend, xs[i].contiguous(), peer))
        if s_recv[i] > 0:
            ops.append(dist.P2POp(dist.irecv, ys[i], peer))
    if ops:
        for r in dist.batch_isend_irecv(ops):
            r.wait()
    out = torch.cat(ys)
    return out[:, 0] if squeeze else out


class A2A(torch.autograd.Function):             # all_to_all 的反向仍是 all_to_all（方向对调）
    @staticmethod
    def forward(ctx, x, s_send, s_recv):
        ctx.s_send, ctx.s_recv = s_send, s_recv
        _rec("all_to_all[ep]", x)
        return _a2a(x, s_send, s_recv)

    @staticmethod
    def backward(ctx, grad):
        _rec("all_to_all[ep]", grad)
        return _a2a(grad.contiguous(), ctx.s_recv, ctx.s_send), None, None


def moe(x, Wr, EX):                             # MoE 层：路由 → 分发 → 本地专家 → 收回
    prob = F.softmax(x @ Wr, dim=-1)            # [T, NE] 路由分数
    sel = prob.argmax(dim=-1)                   # top-1：每个 token 选 1 个专家
    pk = prob.gather(1, sel[:, None])           # [T,1] 选中专家的分数
    perm = torch.argsort(sel // EL, stable=True)    # 按目标卡排序，便于分块
    s_send = torch.bincount(sel // EL, minlength=2).tolist()
    cnt = torch.tensor(s_send, dtype=torch.long, device=dev)
    _rec("all_to_all[ep-meta]", cnt)
    s_recv = _a2a(cnt, [1, 1], [1, 1]).tolist()     # 先互换条数，再发数据
    buf = torch.cat([x[perm], pk[perm], sel[perm][:, None].float()], dim=1)  # 随行带上分数与专家号
    rec = A2A.apply(buf, s_send, s_recv)        # 分发
    xs, ps = rec[:, :H], rec[:, H:H + 1]
    ids = rec[:, H + 1].long() - ep * EL        # 换算成本地专家号
    y = torch.zeros_like(xs)
    for i, (W1, W2) in enumerate(EX):           # 只算发给本地专家的 token
        idx = (ids == i).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            y = torch.index_add(y, 0, idx, F.gelu(xs[idx] @ W1) @ W2)
    back = A2A.apply(y * ps, s_recv, s_send)    # 加权后按原路发回
    return torch.zeros_like(x).index_copy(0, perm, back)   # 恢复原始 token 顺序


def layer(x):
    h = x + attn(ln(x), W["Wq"], W["Wk"], W["Wv"], W["Wo"])
    return h + moe(ln(h), W["Wr"], EX)


def moe_ref(x, Wr, EXF):                        # 单卡参考：稠密 MoE，一个 token 也不多走
    prob = F.softmax(x @ Wr, dim=-1)
    sel = prob.argmax(dim=-1)
    pk = prob.gather(1, sel[:, None])
    out = torch.zeros_like(x)
    for j in range(NE):
        idx = (sel == j).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            W1, W2 = EXF[j]
            out = torch.index_add(out, 0, idx, pk[idx] * (F.gelu(x[idx] @ W1) @ W2))
    return out


def layer_ref(x, wf):
    h = x + attn(ln(x), wf["Wq"], wf["Wk"], wf["Wv"], wf["Wo"])
    return h + moe_ref(ln(h), wf["Wr"], wf["EX"])


# ---------- 前向 + 反向 ----------
comm_reset()
if pp == 0:                                     # 首段：嵌入 + 第 0 层
    h = layer(F.embedding(tok, E))
    act = h.detach().contiguous()               # 边界激活：发给下一段
    _rec("send[pp]", act)
    dist.send(act, group=pp_group, group_dst=1)
else:                                           # 末段：收激活 + 第 1 层 + 输出头
    act = torch.empty(T, H, device=dev)
    _rec("recv[pp]", act)
    dist.recv(act, group=pp_group, group_src=0)
    act.requires_grad_(True)
    h = layer(act)
    loss = F.cross_entropy(h @ E.T, tgt)
comm_report_rank("forward")

if pp == 1:
    loss.backward()
    dA = act.grad.contiguous()                  # 边界梯度：发回上一段
    _rec("send[pp]", dA)
    dist.send(dA, group=pp_group, group_dst=0)
else:
    dA = torch.empty(T, H, device=dev)
    _rec("recv[pp]", dA)
    dist.recv(dA, group=pp_group, group_src=1)
    h.backward(dA)
comm_report_rank("backward")

# 没收到任何 token 的专家（或未参与本段计算的参数）梯度为 None，按 0 计
for q in [E, *W.values()] + [w for pair in EX for w in pair]:
    if q.grad is None:
        q.grad = torch.zeros_like(q)

# step 末三次求和：① embedding 组（嵌入侧 + 输出头侧）② DP 组（稠密参数 + router）③ EDP 组（专家）
_rec("all_reduce[emb]", E.grad)
dist.all_reduce(E.grad, group=emb_group)
for p in [E, W["Wq"], W["Wk"], W["Wv"], W["Wo"], W["Wr"]]:
    _rec("all_reduce[dp]", p.grad)
    dist.all_reduce(p.grad, group=dp_group)
for W1, W2 in EX:
    for p in (W1, W2):
        _rec("all_reduce[edp]", p.grad)
        dist.all_reduce(p.grad, group=edp_group)
comm_report_rank("step 末梯度同步")

# ---------- 参考：单卡把四条车道各算一遍，梯度相加（÷dp 省略，与篇 7 同口径） ----------
gE = torch.zeros_like(E_full)
gW = [{k: torch.zeros_like(wf[k]) for k in ("Wq", "Wk", "Wv", "Wo", "Wr")} for wf in LAY_full]
gX = [[(torch.zeros_like(w1), torch.zeros_like(w2)) for (w1, w2) in wf["EX"]] for wf in LAY_full]
loss_ref = act_ref = dA_ref = None
for j in range(4):
    torch.manual_seed(1000 + j)                 # 复现车道 j 的数据
    tokj = torch.randint(0, V, (T,), device=dev)
    tgtj = torch.randint(0, V, (T,), device=dev)
    Ej = E_full.clone().requires_grad_(True)
    Lj = [{k: wf[k].clone().requires_grad_(True) for k in ("Wq", "Wk", "Wv", "Wo", "Wr")}
          | {"EX": [(a.clone().requires_grad_(True), b.clone().requires_grad_(True))
                    for (a, b) in wf["EX"]]} for wf in LAY_full]
    h0 = layer_ref(F.embedding(tokj, Ej), Lj[0])
    h0.retain_grad()                            # 它就是"边界激活"的参考
    lj = F.cross_entropy(layer_ref(h0, Lj[1]) @ Ej.T, tgtj)
    lj.backward()
    gE += Ej.grad
    for i in range(2):
        for k in ("Wq", "Wk", "Wv", "Wo", "Wr"):
            gW[i][k] += Lj[i][k].grad
        for n in range(NE):
            gr1, gr2 = Lj[i]["EX"][n][0].grad, Lj[i]["EX"][n][1].grad   # 本条车道没选中该专家时为 None，按 0 计
            gX[i][n] = (gX[i][n][0] + (gr1 if gr1 is not None else 0),
                        gX[i][n][1] + (gr2 if gr2 is not None else 0))
    if j == dp:
        loss_ref, act_ref, dA_ref = lj.detach(), h0.detach(), h0.grad


def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-5)


ok_e = same(E.grad, gE)
ok_w = all(same(W[k].grad, gW[pp][k]) for k in ("Wq", "Wk", "Wv", "Wo", "Wr"))
ok_x = all(same(EX[i][0].grad, gX[pp][ep * EL + i][0]) and same(EX[i][1].grad, gX[pp][ep * EL + i][1])
           for i in range(EL))
if pp == 0:
    ok_act, ok_da = same(act, act_ref), same(dA, dA_ref)
    ok = all([ok_act, ok_da, ok_w, ok_x, ok_e])
    print(f"rank{rank}：边界激活对比 {ok_act}，边界梯度对比 {ok_da}，attention/router 梯度对比 {ok_w}，"
          f"专家梯度对比 {ok_x}，embedding 梯度对比 {ok_e} [{'OK' if ok else 'FAIL'}]", flush=True)
else:
    ok_loss = same(loss.detach(), loss_ref)
    ok = all([ok_loss, ok_w, ok_x, ok_e])
    print(f"rank{rank}：loss 对比 {ok_loss}，attention/router 梯度对比 {ok_w}，"
          f"专家梯度对比 {ok_x}，embedding 梯度对比 {ok_e} [{'OK' if ok else 'FAIL'}]", flush=True)

dist.destroy_process_group()