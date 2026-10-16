# exp28_channels.py —— 实验四：固定 channel 数，测大消息耗时（GPU 专属）
# 用法（env 在建组时读取，每组 channel 数需要单独启动一次）：
#   NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 torchrun --nproc_per_node=8 exp28_channels.py
#   NCCL_MIN_NCHANNELS=2 NCCL_MAX_NCHANNELS=2 torchrun --nproc_per_node=8 exp28_channels.py
#   ... 依次换 4/8/16/32
import os
import time

import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验四：channel 数扫描")
assert dev.type == "cuda", "本篇实验都需要 GPU/NCCL"
nc = os.environ.get("NCCL_MAX_NCHANNELS", "auto")

S = 2**30                                # 1 GiB
t = torch.ones(S // 4, device=dev)
for _ in range(3):
    dist.all_reduce(t)
torch.cuda.synchronize()

ts = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(t)
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
ts.sort()
t_med = ts[len(ts) // 2]
busbw = 2 * (world - 1) / world * S / t_med / 1e9

if rank == 0:
    print(f"channels={nc:>4}  1 GiB  中位 {t_med * 1e3:7.2f} ms  busbw ≈ {busbw:.0f} GB/s")
dist.destroy_process_group()
