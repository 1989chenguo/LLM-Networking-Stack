# exp17_ep_imbalance.py —— 实验二：路由偏斜——负载不均如何变成通信不均，容量上限怎么丢 token
# 同一个 MoE 层（4 专家、每卡 2 个、每 token 选 top-2），两种手工构造的路由：
#   uniform：首选、次选都轮转 → 每个专家恰好分到平均数
#   skewed ：3/4 的 token 首选专家 0 → 专家 0、1（都在 rank0 上）远远超载
# 每种路由各跑一次真实 dispatch，看各卡收发字节；再模拟容量上限下的丢弃数量。
# 路由是手工构造的、与随机数无关，所以 GPU 和 CPU 上的输出完全一致。
# GPU 机器上默认 cuda/nccl，其余机器自动退回 cpu/gloo（也可用 --device cpu 强制）。
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验二：路由偏斜")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

E, T, K, H = 4, 64, 2, 256    # 专家数、每卡 token 数、top-k、hidden
EPR = E // world              # 每卡 2 个专家：专家 0、1 在 rank0，专家 2、3 在 rank1

t = torch.arange(T, device=dev)
uniform = torch.stack([t % E, (t + 1) % E], dim=1)                        # 均匀：每个专家一样多
first = torch.where(t % 4 > 0, torch.zeros_like(t), (t // 4) % (E - 1) + 1)
skewed = torch.stack([first, (first + 1) % E], dim=1)                     # 偏斜：3/4 首选专家 0

torch.manual_seed(1000 + rank)         # 两卡不同数据
X = torch.randn(T, H, device=dev)

def dispatch(idx, tag):
    """dropless 分发：打包 → 互换计数 → 寄 token 与专家编号；打印各卡收发情况。"""
    exp_id = idx.reshape(-1)
    tok = torch.arange(T, device=dev).repeat_interleave(K)
    dest = exp_id // EPR
    counts = torch.bincount(dest, minlength=world)          # 我寄给每张卡多少行
    peer = torch.empty(world, dtype=torch.int64, device=dev)
    dist.all_to_all_single(peer, counts)                    # 每张卡寄给我多少行
    ss, rs = counts.tolist(), peer.tolist()
    order = torch.argsort(dest, stable=True)
    recv_buf = torch.empty(sum(rs), H, device=dev)
    dist.all_to_all_single(recv_buf, X[tok[order]].contiguous(), rs, ss)
    ids_recv = torch.empty(sum(rs), dtype=torch.int64, device=dev)
    dist.all_to_all_single(ids_recv, exp_id[order].contiguous(), rs, ss)
    per_expert = torch.bincount(ids_recv, minlength=E)[rank * EPR:(rank + 1) * EPR]
    print(f"rank{rank} [{tag}] 发出 128 KiB（寄给 rank0 {ss[0]} 行、rank1 {ss[1]} 行），"
          f"收到 {recv_buf.numel() * 4 // 2**10} KiB；"
          f"本卡专家各收到 {per_expert.tolist()} 个 token", flush=True)
    return per_expert

def cap_report(per_exp, tag):
    """容量上限 = cf × 平均每专家份数；超出的 token 直接丢弃（Switch 的做法）。"""
    avg = world * T * K // E
    line = f"  [{tag}] "
    for cf in (1.0, 1.25, 1.5, 2.0):
        cap = int(cf * avg)
        dropped = int((per_exp - cap).clamp(min=0).sum())
        line += f"cf={cf:g} 上限{cap} 丢 {dropped} 份（{dropped / (world * T * K):.0%}）  "
    if rank == 0:
        print(line.rstrip(), flush=True)

for tag, idx in [("uniform", uniform), ("skewed", skewed)]:
    per_exp = dispatch(idx, tag)                  # 本卡两个专家各自收到的总数（两卡合计）
    gathered = [torch.empty(EPR, dtype=torch.int64, device=dev) for _ in range(world)]
    dist.all_gather(gathered, per_exp)
    cap_report(torch.cat(gathered), tag)          # 拼成全专家视图，模拟容量丢弃

dist.destroy_process_group()
