# exp21_stream.py —— 实验一：stream 的脾气三连验（GPU 专属，单进程直接跑）
import time

import torch

assert torch.cuda.is_available(), "这个实验要拍 GPU 的队列，请在 GPU 机器上跑"
torch.cuda.init()

CYCLES = 100_000_000  # 每个 kernel 空转的时钟周期数（约几十毫秒，随卡而定）

# ── 第一组：两个 kernel 排进同一条流（当前流）——性质一：流内严格保序
torch.cuda.synchronize()
t0 = time.perf_counter()
torch.cuda._sleep(CYCLES)  # _sleep：PyTorch 测试自用的小 kernel，在 GPU 上空转 N 个周期
torch.cuda._sleep(CYCLES)
torch.cuda.synchronize()
t_serial = time.perf_counter() - t0

# ── 第二组：两个 kernel 分进两条流——性质二：流间允许并行
s = torch.cuda.Stream()              # 从池里领一条新流（§2.3 同一口池子）
torch.cuda.synchronize()
t0 = time.perf_counter()
torch.cuda._sleep(CYCLES)            # 当前流
with torch.cuda.stream(s):           # 这个 with 块里，算子都排进 s
    torch.cuda._sleep(CYCLES)
torch.cuda.synchronize()             # 等所有流清空
t_parallel = time.perf_counter() - t0

print(f"[性质一] 同一条流排两个长 kernel：{t_serial * 1e3:.1f} ms")
print(f"[性质二] 两条流各排一个：        {t_parallel * 1e3:.1f} ms")
print(f"         加速比：{t_serial / t_parallel:.2f}x")

# ── 第三组：跨流读写同一个张量，不画线 vs 画线——性质四：跨流先后只能靠 event
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
# 提前分配所有张量，关键！避免在流内部触发 cudaMalloc
a = torch.zeros(1024, device="cuda")
b = torch.empty_like(a)  # 提前给 b 分配好显存
c = torch.empty_like(a)  # 提前给 c 分配好显存

# 不画线：s1 慢吞吞地写，s2 不等就读
with torch.cuda.stream(s1):
    torch.cuda._sleep(CYCLES)
    a.fill_(1.0)
with torch.cuda.stream(s2):
    b.copy_(a)       # 用 copy_ + mul_ 替代 a*2，不分配新内存
    b.mul_(2)
torch.cuda.synchronize()
print(f"[性质四] 跨流不画线：b[0] = {b[0].item():.0f}（此时应该读到旧值 0）")

# 画线：正常同步版本
ev = torch.cuda.Event()
a.fill_(0.0)
with torch.cuda.stream(s1):
    torch.cuda._sleep(CYCLES)
    a.fill_(1.0)
    ev.record(s1)
with torch.cuda.stream(s2):
    s2.wait_event(ev)
    c.copy_(a)
    c.mul_(2)
torch.cuda.synchronize()
print(f"[性质四] 跨流画线：  c[0] = {c[0].item():.0f}（等到写完再读，对了）")