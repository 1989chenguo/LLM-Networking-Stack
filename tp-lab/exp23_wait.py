# exp23_wait.py —— 实验三：wait() 到底等什么（GPU 上答案会出乎意料）
import time

import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验三：wait() 到底等什么")

# 计算和通信分开给料：计算要够重（让 GPU 队列忙起来），通信要够大（让 gloo 等得出来）
n, iters = (4096, 50) if dev.type == "cuda" else (256, 20)
t = torch.ones(n, n, device=dev) / n
buf = torch.ones(16384, 1024, device=dev)   # 64 MB 通信缓冲

dist.all_reduce(buf)  # 预热：建组在镜头外完成（第 3 篇）

# 先往队列里排一长串计算：GPU 得忙上好一阵，CPU 则当场算完
if dev.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    t = t @ t
t_launch = (time.perf_counter() - t0) * 1e3

h = dist.all_reduce(buf, async_op=True)  # 派单（第 7 篇的手法）

t0 = time.perf_counter()
h.wait()                                 # ← 本篇的主角：它"等"了多久？
t_wait = (time.perf_counter() - t0) * 1e3

t0 = time.perf_counter()
if dev.type == "cuda":
    torch.cuda.synchronize()             # 这才是"CPU 真等 GPU"
t_sync = (time.perf_counter() - t0) * 1e3

if rank == 0:
    print(f"发射 {iters} 个矩阵乘：CPU 花 {t_launch:.2f} ms（队列还在忙）")
    print(f"h.wait()：          CPU 花 {t_wait:.2f} ms")
    print(f"synchronize()：     CPU 花 {t_sync:.2f} ms（队列全部清空）")
dist.destroy_process_group()
