# exp25_topo.py —— 实验一：触发一次建组，配合环境变量导出拓扑与环（GPU 专属）
# 本体与篇 2 的 exp0 等价；观测全部在环境变量里：
#   NCCL_TOPO_DUMP_FILE   导出拓扑 XML（默认只有 rank 0 写文件）
#   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH   打印建组日志
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验一：拓扑文件与环日志")
assert dev.type == "cuda", "本篇实验都需要 GPU/NCCL"

t = torch.tensor([rank + 1.0], device=dev)
dist.all_reduce(t)                       # 第一次使用这个组：触发建组
expected = world * (world + 1) / 2
assert t.item() == expected, f"期望 {expected}，得到 {t.item()}"
if rank == 0:
    print(f"OK：world={world}，建组完成，拓扑文件与日志已生成")
dist.destroy_process_group()
