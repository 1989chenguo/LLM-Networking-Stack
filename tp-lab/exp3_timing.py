# exp3_timing.py —— 实验一：第一次 all_reduce 为什么慢
import time
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验一：第一次为什么慢")
t = torch.ones(1024, 1024, device=dev)   # 4 MB 的 fp32

for i in range(6):
    if dev.type == "cuda":
        torch.cuda.synchronize()         # 计时对齐：等 GPU 把排队的活干完
    t0 = time.perf_counter()
    dist.all_reduce(t)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    if rank == 0:
        print(f"第 {i} 次 all_reduce：{(time.perf_counter() - t0) * 1e3:.2f} ms", flush=True)
dist.destroy_process_group()
