# exp22_two_streams.py —— 实验二：队列证件照 + 对齐的时间轴证据
import json
import torch
import torch.distributed as dist
from mini_tp import lab_init
from torch.profiler import profile, ProfilerActivity

dev, rank, world = lab_init("实验二：队列证件照与对齐证据（最终修正版）")
if dev.type != "cuda":
    raise SystemExit("这个实验要拍 GPU 的任务队列，请用 GPU 机器跑（去掉 --device cpu）")

t = torch.ones(1024, 1024, device=dev)
buf_comm = torch.ones(65536, 1024, device=dev)    # 256MB 通信专用缓冲
world_size = dist.get_world_size()

# 预热：建组 + 大张量通信初始化，消除首次分配开销
for _ in range(3):
    dist.all_reduce(t)
    dist.all_reduce(buf_comm)
    buf_comm.mul_(2)
torch.cuda.synchronize()

# 预热 profiler，避免第一次 with profile 的 CUPTI 初始化进入测量
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    buf_comm.mul_(2)
    torch.cuda.synchronize()

def analyze_trace(prof_path, label):
    """统一分析 trace 并打印结果"""
    events = json.load(open(prof_path))["traceEvents"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    t0 = kernels[0]["ts"]
    print(f"\n{'kernel 名':<46}{'队列':<4}{'起点(µs)':<10}{'终点(µs)'}")
    for e in kernels:
        stream_id = (e.get('args') or {}).get('stream', '?')
        print(f"{e['name'][:44]:<46}{stream_id:<4}{e['ts'] - t0:<10.0f}{e['ts'] + e['dur'] - t0:.0f}")
    
    comms = [e for e in kernels if "ncclDevKernel" in e["name"]]
    eltwise = [e for e in kernels if "elementwise" in e["name"]]
    if len(comms) >= 2 and len(eltwise) >= 4:
        k2, k3 = comms[0], comms[1]
        end2 = k2["ts"] + k2["dur"]
        lo = max(k3["ts"], eltwise[0]["ts"])
        hi = min(k3["ts"] + k3["dur"], eltwise[-1]["ts"] + eltwise[-1]["dur"])
        print(f"\n③ 起点({k3['ts'] - t0:.0f}) ≥ ② 终点({end2 - t0:.0f})？{k3['ts'] >= end2}   ← 对齐 event 的痕迹")
        print(f"③ 与 ④ 的时间轴重叠 ≈ {max(0.0, hi - lo):.0f} µs   ← {label}")
    return kernels


buf_comm.fill_(1.0)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    y = t @ t
    dist.all_reduce(t)                       # ② 同步通信
    h = dist.all_reduce(buf_comm, async_op=True)  # ③ 异步通信
    for _ in range(8):
        buf_comm.mul_(2)                    # ④ 同张量计算 
    h.wait()
    torch.cuda.synchronize()

if rank == 0:
    prof.export_chrome_trace("/tmp/exp22_case1.json")
    analyze_trace("/tmp/exp22_case1.json", "计算通信并行")
    gt = torch.ones_like(buf_comm) * world_size
    for _ in range(8): gt.mul_(2)
    print(f"数值正确性：{torch.allclose(buf_comm, gt, rtol=1e-3)}  ← 计算通信并行，同张量写竞争，结果错误")

dist.destroy_process_group()
