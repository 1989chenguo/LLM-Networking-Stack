# exp4_profiler.py —— 实验二：用 profiler 看 all_reduce 的真身（GPU 专属）
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验二：看见 all_reduce 的真身")
t = torch.ones(1024, 1024, device=dev)       # 4 MB 的 fp32

for _ in range(3):                           # 预热 3 次：建组的几百毫秒发生在第一次，
    dist.all_reduce(t)                       # 让它在镜头外做完，别挡镜头

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU,   # 开录：CPU、GPU 两侧都记
                         ProfilerActivity.CUDA]) as prof:
    dist.all_reduce(t)                       # 镜头内只拍一次干净的通信
    torch.cuda.synchronize()                 # 等 GPU 干完，kernel 才算被拍到

if rank == 0:                                # 只让 rank 0 打印，否则两个进程各打一份
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
dist.destroy_process_group()
