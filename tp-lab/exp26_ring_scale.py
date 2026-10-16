# exp26_ring_scale.py —— 实验二：固定消息大小，时间是否随卡数变化（GPU 专属）
# 分别用 torchrun --nproc_per_node=2 / 4 / 8 各跑一次
import time

import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验二：ring 的时间与卡数")
assert dev.type == "cuda", "本篇实验都需要 GPU/NCCL"

S = 256 * 2**20                       # 256 MiB
t = torch.ones(S // 4, device=dev)    # fp32

for _ in range(5):                    # 预热：建组与首次发射在计时之外
    dist.all_reduce(t)
torch.cuda.synchronize()

ts = []
for _ in range(20):
    torch.cuda.synchronize()          # 计时对齐（第 3 篇的常识）
    t0 = time.perf_counter()
    dist.all_reduce(t)
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
ts.sort()
t_med = ts[len(ts) // 2]
busbw = 2 * (world - 1) / world * S / t_med / 1e9   # 折算成 busbw，GB/s

if rank == 0:
    print(f"world={world}  消息 256 MiB  中位时间 {t_med * 1e3:.2f} ms  busbw ≈ {busbw:.0f} GB/s")
dist.destroy_process_group()
