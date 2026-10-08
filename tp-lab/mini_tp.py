# mini_tp.py —— 第三版：lab_init 双模式入口 + 通信计数器 + mappings 六函数 + 组参数化三函数
import argparse
import os
import torch
import torch.distributed as dist


def lab_init(desc):
    """实验统一入口：解析 --device，初始化进程组，返回 (dev, rank, world)。
    GPU 机器上默认 cuda/nccl；任何笔记本上自动退回 cpu/gloo。"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cpu")
    return dev, rank, world


# ===== 第 6 篇加长（第一批）：通信计数器 + Megatron mappings 四函数 =====
# 语义对齐 megatron/core/tensor_parallel/mappings.py；组写死为 WORLD
# （本篇 world=2，把 WORLD 当 TP 组用；真实版每个函数带 group 参数）。

COMM_LOG = []          # 每次集合通信记一笔：(算子名, 本卡经手字节数)

def _rec(op, t):
    COMM_LOG.append((op, t.numel() * t.element_size()))

def comm_reset():
    COMM_LOG.clear()

def comm_report(tag=""):
    """打印通信账（rank 0 输出，全组清零）：各算子的次数与字节数。"""
    if dist.get_rank() == 0:
        agg = {}
        for op, nbytes in COMM_LOG:
            cnt, b = agg.get(op, (0, 0))
            agg[op] = (cnt + 1, b + nbytes)
        parts = "  ".join(f"{op}×{c}（{b / 2**20:.1f} MiB）" for op, (c, b) in agg.items())
        print(f"[通信账] {tag}：{parts}", flush=True)
    COMM_LOG.clear()


class _CopyToTpRegion(torch.autograd.Function):
    """f：前向恒等（复制），反向 all-reduce —— column 层入口"""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.clone(memory_format=torch.contiguous_format)
        _rec("all_reduce", g)
        dist.all_reduce(g)
        return g


class _ReduceFromTpRegion(torch.autograd.Function):
    """g：前向 all-reduce，反向恒等（复制）—— row 层出口"""
    @staticmethod
    def forward(ctx, x):
        _rec("all_reduce", x)
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToTpRegion(torch.autograd.Function):
    """前向沿最后一维取自己那片，反向 all-gather 拼全"""
    @staticmethod
    def forward(ctx, x):
        world, rank = dist.get_world_size(), dist.get_rank()
        assert x.size(-1) % world == 0, "最后一维必须能被卡数整除"
        return x.chunk(world, dim=-1)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        world = dist.get_world_size()
        _rec("all_gather", grad_output)
        outs = [torch.empty_like(grad_output) for _ in range(world)]
        dist.all_gather(outs, grad_output.contiguous())
        return torch.cat(outs, dim=-1)


class _GatherFromTpRegion(torch.autograd.Function):
    """前向 all-gather 拼全，反向取自己那片"""
    @staticmethod
    def forward(ctx, x):
        world = dist.get_world_size()
        _rec("all_gather", x)
        outs = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(outs, x.contiguous())
        return torch.cat(outs, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        world, rank = dist.get_world_size(), dist.get_rank()
        return grad_output.chunk(world, dim=-1)[rank].contiguous()


def copy_to_tp_region(x):      return _CopyToTpRegion.apply(x)
def reduce_from_tp_region(x):  return _ReduceFromTpRegion.apply(x)
def scatter_to_tp_region(x):   return _ScatterToTpRegion.apply(x)
def gather_from_tp_region(x):  return _GatherFromTpRegion.apply(x)


# ===== 第 6 篇加长（第二批）：序列并行的 f'/g' =====

class _GatherFromSeqRegion(torch.autograd.Function):
    """SP 版 f：前向沿序列维（dim 0）all-gather，反向 reduce-scatter"""
    @staticmethod
    def forward(ctx, x):
        world = dist.get_world_size()
        _rec("all_gather", x)
        outs = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(outs, x.contiguous())
        return torch.cat(outs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        world = dist.get_world_size()
        out = torch.empty(grad_output.size(0) // world, *grad_output.shape[1:],
                          device=grad_output.device, dtype=grad_output.dtype)
        _rec("reduce_scatter", out)
        dist.reduce_scatter_tensor(out, grad_output.contiguous())
        return out


class _ReduceScatterToSeqRegion(torch.autograd.Function):
    """SP 版 g：前向沿序列维 reduce-scatter，反向 all-gather"""
    @staticmethod
    def forward(ctx, x):
        world = dist.get_world_size()
        out = torch.empty(x.size(0) // world, *x.shape[1:],
                          device=x.device, dtype=x.dtype)
        _rec("reduce_scatter", out)
        dist.reduce_scatter_tensor(out, x.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        world = dist.get_world_size()
        _rec("all_gather", grad_output)
        outs = [torch.empty_like(grad_output) for _ in range(world)]
        dist.all_gather(outs, grad_output.contiguous())
        return torch.cat(outs, dim=0)


def gather_from_seq_region(x):       return _GatherFromSeqRegion.apply(x)
def reduce_scatter_to_seq_region(x): return _ReduceScatterToSeqRegion.apply(x)


# ===== 第 10 篇加长（第三批）：进程组参数化的 mappings + 全 rank 通信账 =====
# 第 6 篇的六函数把组写死为 WORLD；组合并行里一张卡同时属于好几个子组，
# 通信必须指明发生在哪个组——这也是 Megatron 真实版的形态（每个函数带
# group 参数）。f/g/gather 三个的语义与第 6 篇逐条对应。通信账按组标签
# 分列，每个 rank 打印自己那本。

class _CopyToRegion(torch.autograd.Function):
    """f：前向恒等（逻辑复制），反向在 group 内 all-reduce —— column 层入口"""
    @staticmethod
    def forward(ctx, x, group, tag):
        ctx.group, ctx.tag = group, tag
        return x

    @staticmethod
    def backward(ctx, grad_output):
        g = grad_output.clone(memory_format=torch.contiguous_format)
        _rec(f"all_reduce[{ctx.tag}]", g)
        dist.all_reduce(g, group=ctx.group)
        return g, None, None


class _ReduceFromRegion(torch.autograd.Function):
    """g：前向在 group 内 all-reduce，反向恒等 —— row 层出口 / 词嵌入出口"""
    @staticmethod
    def forward(ctx, x, group, tag):
        _rec(f"all_reduce[{tag}]", x)
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _GatherFromRegion(torch.autograd.Function):
    """前向在 group 内 all-gather 沿末维拼全，反向取自己那片 —— 输出头出口"""
    @staticmethod
    def forward(ctx, x, group, tag):
        ctx.group = group
        _rec(f"all_gather[{tag}]", x)
        outs = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
        dist.all_gather(outs, x.contiguous(), group=group)
        return torch.cat(outs, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        rank = dist.get_rank(ctx.group)
        out = grad_output.chunk(dist.get_world_size(ctx.group), dim=-1)[rank]
        return out.contiguous(), None, None


def copy_to_region(x, group, tag):       return _CopyToRegion.apply(x, group, tag)
def reduce_from_region(x, group, tag):   return _ReduceFromRegion.apply(x, group, tag)
def gather_from_region(x, group, tag):   return _GatherFromRegion.apply(x, group, tag)


def comm_report_rank(tag=""):
    """comm_report 的全 rank 版：每个进程打印自己那本账，打印后清零。"""
    agg = {}
    for op, nbytes in COMM_LOG:
        cnt, b = agg.get(op, (0, 0))
        agg[op] = (cnt + 1, b + nbytes)
    def fmt(b):
        return f"{b / 2**20:.1f} MiB" if b >= 2**20 else f"{b / 2**10:.1f} KiB"
    parts = "  ".join(f"{op}×{c}（{fmt(b)}）" for op, (c, b) in sorted(agg.items()))
    print(f"[通信账] rank{dist.get_rank()} {tag}：{parts}", flush=True)
    COMM_LOG.clear()
