# exp27_proto_sweep.py —— 实验三：同一组尺寸下三种协议的耗时（GPU 专属）
# 用法：
#   NCCL_PROTO=LL     torchrun --nproc_per_node=2 exp27_proto_sweep.py
#   NCCL_PROTO=LL128  torchrun --nproc_per_node=2 exp27_proto_sweep.py
#   NCCL_PROTO=Simple torchrun --nproc_per_node=2 exp27_proto_sweep.py
#   （不设 NCCL_PROTO 的一次运行是对照组：NCCL 自动选择）
import os
import time

import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验三：协议扫描")
assert dev.type == "cuda", "本篇实验都需要 GPU/NCCL"
proto = os.environ.get("NCCL_PROTO", "auto")

if rank == 0:
    print(f"协议 = {proto}（环境变量在建组时读取，每次运行只测一种）")

for exp in range(12, 29, 2):                 # 4 KiB, 16 KiB, ..., 256 MiB
    S = 2**exp
    t = torch.ones(S // 4, device=dev)       # fp32
    for _ in range(5):
        dist.all_reduce(t)                   # 预热
    torch.cuda.synchronize()
    ts = []
    iters = 50 if S <= 2**20 else 10
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(t)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    if rank == 0:
        print(f"  {S / 2**20:8.2f} MiB   中位 {ts[len(ts) // 2] * 1e6:9.1f} µs")
    del t
    torch.cuda.empty_cache()
dist.destroy_process_group()
