# exp2_primitives.py —— 实验二：把集合通信原语各跑一遍，看数据怎么流
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验二：集合通信原语")
me = float(rank + 1)          # 每个 rank 的"身份数"：1, 2, ..., world

def report(name, t):
    """把各 rank 手里的 t 汇总到 rank 0，打印全局视图。"""
    outs = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(outs, t.reshape(1))
    if rank == 0:
        cells = "  ".join(f"r{i}={o.item():.0f}" for i, o in enumerate(outs))
        print(f"{name:<15} {cells}", flush=True)

report("初始", torch.tensor([me], device=dev))

# ① broadcast：src 的值覆盖所有人 —— 一人说，大家听
t = torch.tensor([99.0 if rank == 0 else 0.0], device=dev)
dist.broadcast(t, src=0)
report("broadcast", t)

# ② all_reduce：求和，人人有份
t = torch.tensor([me], device=dev)
dist.all_reduce(t)            # 默认 op=SUM
report("all_reduce", t)

# ③ reduce：求和，但只有 dst 拿结果
t = torch.tensor([me], device=dev)
dist.reduce(t, dst=0)
if rank == 0:
    print(f"reduce          只有 r0 拿到 {t.item():.0f}；其他 rank 手里的值不作数", flush=True)

# ④ all_gather：收集所有人的值拼成列表，人人有份
t = torch.tensor([me], device=dev)
outs = [torch.empty_like(t) for _ in range(world)]
dist.all_gather(outs, t)
if rank == 0:
    print(f"all_gather      每人手里都是完整列表 {[int(o.item()) for o in outs]}", flush=True)

# ⑤ reduce_scatter：先求和，再切片分发 —— 每人只留一片
t = torch.tensor([me * (10 ** i) for i in range(world)], device=dev)  # 每人贡献 world 个数
out = torch.empty(1, device=dev)
dist.reduce_scatter_tensor(out, t)     # 逐位求和后，第 i 片发给 rank i
report("reduce_scatter", out)

# ⑥ send / recv：点对点 —— 流水线并行的语言
if rank == 0:
    t = torch.tensor([777.0], device=dev)
    dist.send(t, dst=1)
elif rank == 1:
    t = torch.empty(1, device=dev)
    dist.recv(t, src=0)
    print(f"send/recv       r1 收到 r0 寄来的 {t.item():.0f}", flush=True)

dist.destroy_process_group()
