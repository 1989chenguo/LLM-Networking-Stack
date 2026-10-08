# exp1_groups.py —— 实验一：子组。"谁和谁通信"是可以指定的
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验一：子组")
assert world == 4, "请用 torchrun --nproc_per_node=4 运行"

g01 = dist.new_group([0, 1])   # 注意：所有进程都要调用 new_group，哪怕自己不进组
g23 = dist.new_group([2, 3])

t = torch.tensor([rank + 1.0], device=dev)
if rank in (0, 1):
    dist.all_reduce(t, group=g01)    # 组内求和：1 + 2
elif rank in (2, 3):
    dist.all_reduce(t, group=g23)    # 组内求和：3 + 4

print(f"rank {rank}: 组内 all_reduce = {t.item()}", flush=True)
dist.destroy_process_group()
