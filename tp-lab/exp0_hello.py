# exp0_hello.py —— 环境自检：N 个进程互相看见对方，并完成一次 all_reduce
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("环境自检")
if dev.type == "cuda":
    print(f"rank {rank}/{world}：{torch.cuda.get_device_name(dev)} @ {dev}", flush=True)
else:
    print(f"rank {rank}/{world}：CPU（gloo 后端）", flush=True)

t = torch.tensor([rank + 1.0], device=dev)   # 每个 rank 拿一个不同的数
dist.all_reduce(t)                           # 求和：1+2+...+world
expect = world * (world + 1) / 2
status = "OK" if t.item() == expect else "FAIL"
print(f"rank {rank}: all_reduce 结果 = {t.item()}（应为 {expect}）[{status}]", flush=True)
dist.destroy_process_group()
