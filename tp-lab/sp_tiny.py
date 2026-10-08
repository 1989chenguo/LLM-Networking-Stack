# sp_tiny.py —— 6.3 口算实例的验证脚本（CPU / gloo，2 进程）
# 运行：torchrun --nproc_per_node=2 sp_tiny.py
import torch
import torch.distributed as dist
import torch.nn.functional as F

dist.init_process_group("gloo")
rank = dist.get_rank()

def log(m):
    print(f"rank{rank} | {m}", flush=True)

# 残差流上的张量 h1：4 个 token，每个 2 维（2.6 例子的延续）
h1_full = torch.tensor([[1., 2.],
                        [3., 0.],
                        [2., 2.],
                        [4., 1.]])

dist.barrier()
# ===== ① TP 模式：两卡拿着一模一样的完整 h1 =====
ln_full = F.layer_norm(h1_full, (2,))
log(f"[TP] 我存完整 4 行 h1；本地算 LN = {ln_full.tolist()}")
log("[TP] 注意：对端卡算出的 LN 和我逐位相同——同样的激活存了两份")
dist.barrier()

# ===== ② SP 模式：每卡只存、只算 2 行 =====
sl = slice(rank * 2, (rank + 1) * 2)
h1_loc = h1_full[sl]
ln_loc = F.layer_norm(h1_loc, (2,))
log(f"[SP] 我只持有 token {sl.start}~{sl.stop - 1}：{h1_loc.tolist()}")
log(f"[SP] 本地 LN = {ln_loc.tolist()}，和完整版第 {sl.start}~{sl.stop - 1} 行一致？"
    f"{torch.allclose(ln_loc, ln_full[sl])}（零通信）")
dist.barrier()

# ===== ③ f'：进 block 前 all-gather，拼回完整序列 =====
outs = [torch.empty_like(ln_loc) for _ in range(2)]
dist.all_gather(outs, ln_loc)
a = torch.cat(outs, dim=0)
log(f"[SP] f' all-gather 后 = {a.tolist()}")
log(f"[SP] 拼回了完整 LN 输出？{torch.allclose(a, ln_full)}")
dist.barrier()

# ===== ④ block 内部各卡算部分和（伪造），g'：reduce-scatter =====
P = torch.arange(8, dtype=torch.float32).reshape(4, 2) * (rank + 1)
log(f"[SP] 假设 block 出口我的部分和 P_{rank} = {P.tolist()}")
out = torch.empty(2, 2)
dist.reduce_scatter_tensor(out, P)          # 求和 + 只留自己那 2 行
log(f"[SP] g' reduce-scatter 后我只留 {out.tolist()}"
    f"（= (P₀+P₁) 的第 {sl.start}~{sl.stop - 1} 行）")

dist.destroy_process_group()
