# 数据并行与 FSDP：通信重叠与显存分片

> 连载第 7 篇 · 并行策略层
> 
> 上一篇分析了张量并行的通信量：每层 4 次 all-reduce，前向出口 2 次、反向入口 2 次，由计算依赖决定，无法避免也无法合并。篇末练习留了一个问题：这些通信能不能**与计算重叠**？——TP 的答案是能重叠一点（层与层之间），但很有限；真正把重叠做到位的是本篇的主角：**数据并行（DP）**。
> 
> DP 的梯度 all-reduce 特性不同：可以分桶、可以异步发起，还能整体与 backward 计算并行。第 3 篇介绍的 Work 句柄（`async_op=True`）在本篇真正派上用场，你会看到通信如何被计算遮住。之后我们再看 DP 的另一项开销——显存：ZeRO 系列的思路是把优化器状态、梯度、参数按卡数切开分存，本篇先跟切得最狠的 ZeRO-3——也就是 FSDP——走到底，用实验称一称"通信换显存"的时间代价；最后再回头补全 ZeRO 的另外两档：如果 FSDP 多付的通信你接受不了，ZeRO-1、ZeRO-2 是不加通信的折中，同样用代码拆开、用实验对比。

**本篇的四个问题**

1. DP 每 step 一次的梯度 all-reduce 不可避免——但发起的时机有自由度。怎么让它与 backward 重叠？
2. 分桶、异步、重叠这三个手段，各自为什么成立？
3. FSDP（ZeRO-3）把每卡显存除以 t，通信多付出了多少？为什么这部分通信也能重叠？
4. 这多付的通信如果接受不了，还有没有别的选项？ZeRO-1、ZeRO-2 在代码上怎么实现，和 FSDP 差在哪几行？

**动手指南**：本篇有四个实验脚本（exp10~exp13），运行方式照旧：

```bash
torchrun --nproc_per_node=2 脚本名.py
```

---

## 一、DP 的通信特点：每 step 一次大消息，和 TP 完全不同

数据并行的语义一句话说完：模型完整复制 t 份，每张卡喂不同的数据，各自算出梯度，然后把 t 份梯度**求和（平均）**，每卡一份，再各自更新——因为各卡的平均梯度相同，更新完权重仍然一致。

通信需求按第 1 篇的统计口径：每个训练 step 的末尾一次，消息是**全体梯度**，GB 级。和 TP 放在一起对比，差异很直观：

|  | TP（上一篇） | DP（本篇） |
| --- | --- | --- |
| 频次 | 每层 4 次 | 每 step 1 次（分桶后变成很多次，但总量不变） |
| 单条消息 | s·b·h，KB~MB 级 | 全体梯度，GB 级 |
| 就绪时机 | 由层内计算依赖决定 | backward 中**逐层倒序就绪** |
| 对什么敏感 | 延迟，必须机内 | 带宽，可以跨机 |
| 能否与计算重叠 | 无法避免，也无法合并 | **可以——本篇的主要内容** |

关键在"就绪时机"那一行。TP 的通信发生在层**内部**（行切出口、列切入口），前后被矩阵乘固定住，时机没有调整的余地。DP 的通信对象——梯度——却是在 backward 的推进过程中**一层一层陆续算出来的**：backward 从最后一层往第一层走，第 6 层的梯度先就绪，第 5 层随后，最后才是第 1 层。

这个就绪顺序给了 DP 一个 TP 没有的自由度：**先就绪的梯度可以先发出**。再加上 DP 要的是带宽而不是延迟（消息足够大时带宽利用率高，第 5 篇 `ib_write_bw` 的拐点图已经看过），把梯度凑成足够大的消息发出去，通信就有了与计算并行的余地。

先看不做重叠的朴素实现。最直白的 DP 训练循环：

```text
[ forward ][ backward 全部算完 ][ 逐层 all_reduce 梯度，逐个等完 ][ optimizer.step ]
                                                  ▲
                                     通信时间在这里完全暴露，GPU/CPU 空等
```

这段流程的问题很直观：backward 结束到参数更新之间，有一段纯粹的等待。注意"逐层"两个字容易误读——这个通信循环的起点在 backward **结束之后**：梯度虽然在 backward 推进时已经逐层就绪，朴素版却把它们全部攒到最后才发。字节数不能少——梯度平均是数学必需——但发起的时机可以调整。下面三个手段，就是把通信从"backward 之后"挪进"backward 之中"。

## 二、三个手段：分桶、异步、重叠

### 2.1 分桶：桶是"就绪"的最小单位

一个模型的参数张量成百上千（每层有权重矩阵，还有很多 bias 这样的小张量），如果每个张量单独发一次 all-reduce，通信次数会非常多，而且每条消息太小、带宽利用率低（第 5 篇讨论过小消息的问题）。所以第一步是**分桶**：把一组参数的梯度**拼（cat）成一个连续的大缓冲区**，一次 all_reduce 发出去。

桶怎么划？教学版最自然的划法是**一层一桶**：某一层 W 和 b 的梯度拼成一个缓冲区，一次发出。8 KB 的 bias 梯度和权重拼在一起发，不单独发。

真实实现（PyTorch 的 DDP）也是这个思路，只是桶按**字节**划：默认每桶约 25 MB（`bucket_cap_mb`），跨层拼桶；它在反向中通过梯度钩子（`param.register_hook`）自动完成"哪个就绪哪个入桶、桶满即发"，第一步跑完后还会按梯度实际大小重建一次桶（`rebuild_buckets`）。我们手写的"一层一桶"是它的最小骨架，原理一致。

### 2.2 异步：Work 句柄

第 3 篇说过，c10d 的每次集合通信都会返回一个 **Work 句柄**——当时只用它做看门狗的超时检测。这个句柄的另一半功能本篇才真正用到：`async_op=True`。

```python
handle = dist.all_reduce(bucket, async_op=True)   # 发起后立即返回，CPU 继续执行
# ……这里 CPU 可以立刻去做别的事……
handle.wait()                                     # 等待通信完成
```

对比默认的同步调用：同步版在返回前一直等通信完成，CPU 停在原地；异步版把"发起"和"等待"拆开，中间这段时间可以安排别的工作。

需要注意：**异步本身不省时间**。如果发起后立刻 wait，总时间和同步调用一模一样。省时间的关键在于中间这段时间被填上了什么——也就是下一节的内容。

### 2.3 重叠：用计算填满等待

把前两个手段接起来，重叠就自然出现了：

```text
backward 计算到第 i 层：梯度就绪
   ├─ 拼桶、异步发起               ← 通信开始
   └─ 立刻回头算第 i-1 层           ← 计算与通信同时进行
        ├─ 第 i-1 层梯度就绪 → 拼桶、发起
        └─ ……直到第 1 层
全部发起完毕 → 统一 wait → 更新参数
```

backward 从第 6 层走向第 1 层，每算完一层就把这一桶发出去，通信和计算**同时进行**：CPU/GPU 在算第 i-1 层的反向，网卡/NVLink 在传第 i 层的梯度。

这里有两个容易误解的点值得说清。**第一**，代码里没有"往 GPU 塞 kernel"的语句——**每次算子调用本身就是一次塞入**：执行 `dZ @ Ws[i].T` 时，CPU 把这个 kernel 推进当前 stream 的队列就立即返回，GPU 在后台按队列顺序执行，异步发射是默认行为（本机 gloo 模式下没有 GPU，计算直接在 CPU 上进行，但结构相同）。**第二**，重叠不需要自己再开一个线程：计算和通信本来就在两个执行单元上——计算进当前 stream，NCCL 通信进进程组自己的内部 stream（第 3 篇讲过 stream 的并行队列），两者之间的依赖由 CUDA event 自动维护；gloo 则是进程组里的后台通信线程。两个执行单元在引擎里已经存在，上面那个单循环要决定的只有**发射顺序**：第 i 层梯度一算完就把通信塞进队列，然后不停顿地继续算第 i-1 层。至于 stream 能开多少、优先级怎么给、什么时候需要手动管理，留到高阶章节再展开。

对比两张时间线，"重叠"就很具体了：

```text
朴素版：  [第6层][第5层]...[第1层] │ [ar₁][ar₂]...[ar₆] [update]
           ←── backward 计算 ──→   │ ←─ 通信全部堆在这里 ─→
                                    ↑ 这段时间没有任何计算在进行，设备干等通信

重叠版：  [第6层][第5层]...[第1层]
               [ar₆][ar₅]...[ar₁]   ← 第 i 层一算完就异步发出，
                                        通信和第 i-1 层的计算同时进行
```

三个要点各记一句：**分桶**决定"哪些梯度一起发"；**异步**把"发起"和"等待"拆开；**重叠**把中间的等待时间用来计算——省的不是字节数，是时间。下一节用实验验证。

## 三、实验一：DP 梯度通信的重叠（exp10）

写一个 6 层 MLP，手写反向（反向是我们自己的循环，才能精确控制"哪一层梯度什么时候就绪、什么时候发出"——这也是理解 DDP 原理的最好角度）。新建 `exp10_dp_overlap.py`：

```python
# exp10_dp_overlap.py —— 实验一：DP 梯度同步——朴素阻塞 vs 分桶异步重叠
# 同样的通信量，两种做法：backward 算完再统一同步（通信暴露），
# 还是梯度一就绪就异步发起、与后续层的反向计算重叠。
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import lab_init, _rec, comm_reset, comm_report

dev, rank, world = lab_init("实验一：DP 通信与 backward 重叠")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

torch.manual_seed(0)
L, H, T = 6, 1024, 256          # 层数、宽度、token 数
lr = 0.01
X = torch.randn(T, H, device=dev)     # 两卡数据相同（本实验只对通信计时；
Y = torch.randn(T, H, device=dev)     # DP 的"不同数据"语义在 exp11 里对比验证）

def make_model():
    torch.manual_seed(42)       # 两种模式从同一份权重出发，最后互相校验
    Ws = [torch.randn(H, H, device=dev) / H**0.5 for _ in range(L)]
    bs = [torch.randn(H, device=dev) * 0.02 for _ in range(L)]
    return Ws, bs

def gelu_prime(z):
    # gelu(x) = x·Φ(x)；导数 = Φ(x) + x·φ(x)
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def forward(Ws, bs):
    A = X
    As, Zs = [A], []
    for i in range(L):
        z = A @ Ws[i] + bs[i]
        Zs.append(z)
        A = z if i == L - 1 else F.gelu(z)
        As.append(A)
    return As, Zs

def update(Ws, bs, grads):
    for i in range(L):
        n = Ws[i].numel()
        Ws[i] -= lr * grads[i][:n].view_as(Ws[i])
        bs[i] -= lr * grads[i][n:]

def step_naive(Ws, bs):
    """朴素版：整个反向先算完，再逐层阻塞同步 —— 通信时间完全暴露。"""
    As, Zs = forward(Ws, bs)
    loss = ((As[-1] - Y) ** 2).mean()
    dA = 2 * (As[-1] - Y) / As[-1].numel()   # MSE 的导数
    grads = [None] * L
    for i in reversed(range(L)):             # 反向：梯度按层倒序就绪
        dZ = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = torch.cat([(As[i].T @ dZ).reshape(-1),
                              dZ.sum(0).reshape(-1)]).contiguous()
        dA = dZ @ Ws[i].T
    for i in range(L):                       # ← 通信从这里才开始，且逐层阻塞等待
        _rec("all_reduce", grads[i])
        dist.all_reduce(grads[i])            # 阻塞：发起并等待完成
    update(Ws, bs, grads)
    return loss

def step_overlapped(Ws, bs):
    """重叠版：每一层的梯度一算出来就异步发起通信，然后继续算上一层。"""
    As, Zs = forward(Ws, bs)
    loss = ((As[-1] - Y) ** 2).mean()
    dA = 2 * (As[-1] - Y) / As[-1].numel()
    grads, handles = [None] * L, []
    for i in reversed(range(L)):
        dZ = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = torch.cat([(As[i].T @ dZ).reshape(-1),
                              dZ.sum(0).reshape(-1)]).contiguous()
        _rec("all_reduce", grads[i])
        handles.append(dist.all_reduce(grads[i], async_op=True))  # ← 异步发起后立即返回
        dA = dZ @ Ws[i].T                    # 继续算上一层（通信同时进行）
    for h in handles:
        h.wait()                             # 全部发起后，统一等待完成
    update(Ws, bs, grads)
    return loss

def time_it(fn, Ws, bs, n=5):
    fn(Ws, bs)                               # 预热：建组开销不计入计时
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(Ws, bs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts

Ws_n, bs_n = make_model()
Ws_o, bs_o = make_model()

comm_reset()
naive_ms = time_it(step_naive, Ws_n, bs_n)
comm_report("naive")

comm_reset()
ovl_ms = time_it(step_overlapped, Ws_o, bs_o)
comm_report("overlapped")

if rank == 0:
    for name, ts in [("naive", naive_ms), ("overlapped", ovl_ms)]:
        steps = "  ".join(f"{t:.1f}" for t in ts)
        print(f"{name:<11} 每 step：{steps} ms", flush=True)
    ok = all(torch.allclose(a, b, rtol=1e-4, atol=1e-5)
             for a, b in zip(Ws_n + bs_n, Ws_o + bs_o))
    print(f"两种方式训练 5 步后权重一致：{ok}（通信量相同、耗时不同）", flush=True)

if os.environ.get("TRACE"):                  # TRACE=1 导出 profiler 时间线（GPU 上最清晰）
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step_overlapped(Ws_o, bs_o)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    prof.export_chrome_trace(f"trace_rank{rank}.json")
    if rank == 0:
        print("已导出 trace_rank0.json：用 chrome://tracing 打开，观察通信 kernel 的位置", flush=True)

dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=2 exp10_dp_overlap.py
```

预期输出（计时数字以实测为准，通信量是重点）：

```text
[通信账] naive：all_reduce×36（144.1 MiB）
[通信账] overlapped：all_reduce×36（144.1 MiB）
naive       每 step：47.4  49.7  39.5  49.5  41.7 ms
overlapped  每 step：37.4  33.8  38.5  31.6  37.5 ms
两种方式训练 5 步后权重一致：True（通信量相同、耗时不同）
```

输出有三行，对应三个结论：

1. **通信量完全一致**：`all_reduce×36（144.1 MiB）`——6 步 × 6 层，每层一桶 ≈4 MiB。两种模式发送的字节数、次数完全相同。**通信量没有变，变的只是发起的时机。**
2. **时间更少**：overlapped 每 step 稳定更快。本机是 gloo/CPU（通信线程和计算线程竞争同一批 CPU 核，重叠的收益会打折扣，计时可能波动）；在云上 GPU 机器上，NCCL 通信走独立 stream、NVLink 传输不占计算单元，差距会更明显。
3. **正确性不受影响**：两种方式训练 5 步后权重逐位一致——异步和重叠只是调度方式的改变，数学结果不变。

最后一个观测工具：`TRACE=1` 重跑上面的命令，会导出 `trace_rank0.json`。用 Chrome 打开 `chrome://tracing` 拖进去（GPU 机器上效果最清晰），你会看到时间线里那 6 个 all_reduce 的 kernel 不是集中在最后排成一列，而是分别与某一层的 backward 计算重叠在一起。gloo 模式下通信发生在 CPU 线程里，CUDA 时间线上没有 NCCL kernel，这个"缺席"本身也是证据：通信发生的位置变了。

## 四、另一项开销：显存

通信的重叠解决了，DP 还有一项没有考虑的开销。回头看每个训练 step 每张卡上要存什么（混合精度训练，业界标准配置）：

| 存什么 | 精度 | 每参数字节 |
| --- | --- | --- |
| 参数 | bf16 | 2 B |
| 梯度 | bf16 | 2 B |
| fp32 主权重（优化器维护的精确副本） | fp32 | 4 B |
| Adam 优化器状态（m、v 两份） | fp32 | 8 B |
| **合计** |  | **16 B** |

DP 的模型复制是**全量复制**：每张卡都存完整的 4 份。乘一下就是著名的"显存墙"：

| 模型 | 每卡 DP 显存（16 B/param） | 单卡 80 GB 装得下？ |
| --- | --- | --- |
| 7B | 112 GB | 装不下 |
| 70B | 1.1 TB | 差一个数量级 |
| 175B | 2.8 TB | 差两个数量级 |

通信可以重叠，显存没法重叠——数据必须存在某张卡的显存里才能参与计算（第 1 篇的基本约束）。但 DP 的复制存在明显的冗余：梯度一规约，t 张卡手里的梯度就是同一份；梯度相同、更新规则相同，优化器状态和权重自然也只是同一份东西的 t 个拷贝。ZeRO（Zero Redundancy Optimizer）系列做的事，就是把这三样——**优化器状态、梯度、参数**——按卡数 t 切开分存，每卡只留 1/t。

切到哪一档为止，其实有三种切法（ZeRO-1/2/3）：省显存逐级递增，要付的通信也递增。不过先别急着看分档表——本篇先把切得最狠的那档走到底：**ZeRO-3，连参数一起切，在 PyTorch 里的名字叫 FSDP**（Fully Sharded Data Parallel）。等 §七 实测完 FSDP 和纯 DP 的时间差，§八 再回头把三档摆全——如果 FSDP 多付的通信你接受不了，前两档正是为这种情况准备的。

把 ZeRO-3 翻译成一次训练 step 里每张卡的实际动作，一共三步：

1. **常驻只存分片**：每卡只保存 1/t 的参数分片；整个 step 走完后，手里也只有 1/t 的梯度分片和 1/t 的优化器状态。全量参数在任何时候、任何一张卡上都不存在。
2. **用到哪层拼哪层**：计算推进到某个 block 时，用 all-gather 把 t 片参数拼回全量，这一层算完立刻释放。全量只在计算该层的这段时间里短暂存在。
3. **梯度同样切片**：该层反向先算出本卡数据的完整梯度，再 reduce-scatter 跨卡求和，每卡只留下属于自己的 1/t 切片，交给优化器更新本卡分片。

这三步里藏着理解 FSDP 显存的关键，值得单独说清：**省下来的是常驻显存**——每卡从 16 B/param 降到 16/t B/param，这是跨 step 一直占着、一刻不能少的那份；**付出的是瞬时峰值**——每个 block 计算期间，显存里会短暂多出一份全量参数（反向期间还多一份全量梯度）。由于各 block 顺序计算、用完即释放，PyTorch 的缓存分配器会把这份空间回收、交给下一个 block 复用：需要预留的不是"整个模型的全量"，只是"一个 block 的全量"。这也正是 FSDP 以 block 为单元的原因——单元切多大，直接决定这个瞬时峰值的上限。用一张时间线概括每卡显存的构成：

```text
常驻（全程占用）：分片参数 + 分片梯度 + 分片优化器状态 = 16/t B/param ━━━━━━━━━━
瞬时（各层复用）：          [全量·block i][全量·block i+1][全量·block i+2] …
```

常驻显存 ÷t，换来每 step 多付约 1P 的通信。这笔通信具体是多少、为什么也能与计算重叠，是下一节的内容。

## 五、FSDP 的做法：把 all-reduce 拆成两半

第 2 篇原语表里 all_reduce 有一条脚注，本篇是第三次用到：**ring 实现里，all_reduce = reduce-scatter + all-gather**。序列并行拆过一次（上一篇：SP 把两段拆开各守半个边界），FSDP 再拆一次——拆出来的两半各有用途：

- **参数**：block 的计算需要完整参数。做法：**all-gather**——把 t 片参数拼回全量，用完即释放，不长期保存。
- **梯度**：DP 的原有需求，t 份数据梯度要求平均。做法：**reduce-scatter**——t 份完整梯度求和，每卡只保留自己那一片。

对偶关系（上一篇 2.2 的表）在这里原样复用：all-gather（拼接）的反向是切片，reduce-scatter（求和+切片）的反向是 all-gather——FSDP 的前向和反向同样互为对偶。唯一的新成员是参数上的 all-gather，它服务的是显存：**为了不常驻全量参数，宁可每次临时拼**。

一个 block 在 FSDP 下的完整生命周期：

```text
前向：all-gather 参数（拼全）→ 本地计算 → 释放全量，只留本卡那片分片
反向：all-gather 参数（再拼一次——前向没留，只能重拼）→ 本地计算
      → 梯度 reduce-scatter（求和 + 只留本卡那片）
```

注意哪些是瞬时的：拼出的全量参数、反向中 reduce-scatter 之前的完整梯度，算完即释放；常驻的只有本卡那 1/t 分片。

**为什么反向要再拼一次参数？**把前向拼全的参数用 `ctx.save_for_backward` 存下来，反向不就能直接用？——可以，但那等于每张卡常驻全量参数，显存立刻从 ZeRO-3 退回 ZeRO-2，切分就白做了。FSDP 的选择是：ctx 里只保存 1/t 的分片，反向需要全量时重新 all-gather。省显存、费通信——这笔交换是否划算，算一下就清楚。

**通信量**（每 step、每卡，记参数字节为 P，忽略 (t-1)/t 系数）：

| 方案 | 干什么 | 每 step 字节 |
| --- | --- | --- |
| 纯 DP | all-reduce 全部梯度 | ≈ 2P |
| FSDP | 前向 AG 参数 + 反向 AG 参数 + 梯度 RS | ≈ **3P** |

3P 对 2P：**FSDP 每 step 多付 50% 的通信**。值不值？看换回什么：每卡显存从 16 B/param 降到 16/t B/param——t=64 时 175B 模型从 2.8 TB 降到 44 GB，一张卡放得下。用 1.5 倍通信换 t 倍显存，在显存墙的约束下几乎总是划算的。

而且这部分多出的通信**同样能重叠**：FSDP 的通信天然按层发生，第 i 层 backward 计算时，完全可以预取（prefetch）第 i-1 层的参数——和 DP 分桶重叠是同一个道理。真实 FSDP 就是这么做的（`backward_prefetch`）。预取的代价是：当前层与下一层的全量参数会短暂共存，瞬时峰值要按 1~2 个 block 全量估算——这就是上一节"单元切多大"需要预留的余量。还有一个附带的好处：梯度 reduce-scatter 之后，每卡只有 1/t 的梯度要进优化器，优化器状态本来就只有 1/t——更新环节没有冗余。

## 六、实验二：手写 mini-FSDP（exp11）

分析到这里，再用实验验证一遍。新建 `exp11_fsdp.py`：

```python
# exp11_fsdp.py —— 实验二：mini-FSDP——参数 all-gather、梯度 reduce-scatter，与单卡对比
# 每张卡只存 1/t 的参数分片；使用时 all-gather 拼全（用完即释放），
# 梯度 reduce-scatter 求和、只留本卡那一片。两卡使用不同数据 —— 这是 DP 的本义。
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mini_tp import lab_init, _rec, comm_reset, comm_report

dev, rank, world = lab_init("实验二：mini-FSDP")
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

torch.manual_seed(0)                     # 两卡同种子 → 同一份"完整权重"
H, T = 1024, 256
W1_full = torch.randn(H, 4 * H, device=dev) / H**0.5
W2_full = torch.randn(4 * H, H, device=dev) / (4 * H) ** 0.5
torch.manual_seed(1000 + rank)           # 两卡不同数据 —— 真正的 DP
X = torch.randn(T, H, device=dev)
target = torch.randn(T, H, device=dev)

class _FsdpMatmul(torch.autograd.Function):
    """mini-FSDP 的一次矩阵乘：参数按行分片存储，使用时 all-gather 拼全；
    反向重新拼一次参数，梯度 reduce-scatter 求和、只留本卡那一片。"""

    @staticmethod
    def forward(ctx, x, w_shard):
        world = dist.get_world_size()
        _rec("all_gather", w_shard)
        outs = [torch.empty_like(w_shard) for _ in range(world)]
        dist.all_gather(outs, w_shard.contiguous())
        w_full = torch.cat(outs, dim=0)      # 拼全参数（用完即释放，不保留）
        ctx.save_for_backward(x, w_shard)    # ctx 只保存分片，不保存全量
        return x @ w_full

    @staticmethod
    def backward(ctx, grad_output):
        x, w_shard = ctx.saved_tensors
        world = dist.get_world_size()
        _rec("all_gather", w_shard)
        outs = [torch.empty_like(w_shard) for _ in range(world)]
        dist.all_gather(outs, w_shard.contiguous())
        w_full = torch.cat(outs, dim=0)      # 参数再拼一次 —— 前向没保存，只能重拼
        grad_x = grad_output @ w_full.T
        grad_w_full = x.T @ grad_output      # 本卡数据的完整梯度……
        out = torch.empty_like(w_shard)
        _rec("reduce_scatter", out)
        dist.reduce_scatter_tensor(out, grad_w_full.contiguous())  # ……求和后只留本卡那一片
        return grad_x, out

def fsdp_matmul(x, w_shard): return _FsdpMatmul.apply(x, w_shard)

def same(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-4)

# ---- 参考 loss：本卡数据 + 完整权重，等价于单卡计算 ----
Xr = X.clone().requires_grad_(True)
W1r = W1_full.clone().requires_grad_(True)
W2r = W2_full.clone().requires_grad_(True)
loss_ref = ((F.gelu(Xr @ W1r) @ W2r - target) ** 2).mean()

# ---- 参考梯度：两卡数据的梯度之和（DP 的求和语义；训练时的 ÷world 这里省略） ----
g1 = torch.zeros_like(W1_full)
g2 = torch.zeros_like(W2_full)
for j in range(world):
    torch.manual_seed(1000 + j)          # 复现第 j 张卡的数据
    Xj = torch.randn(T, H, device=dev)
    Yj = torch.randn(T, H, device=dev)
    W1j = W1_full.clone().requires_grad_(True)
    W2j = W2_full.clone().requires_grad_(True)
    ((F.gelu(Xj @ W1j) @ W2j - Yj) ** 2).mean().backward()
    g1 += W1j.grad
    g2 += W2j.grad

# ---- FSDP：每卡只持有 1/world 的参数分片（沿行切，便于梯度分片） ----
rows1, rows2 = H // world, (4 * H) // world
W1 = W1_full[rank * rows1:(rank + 1) * rows1].clone().requires_grad_(True)
W2 = W2_full[rank * rows2:(rank + 1) * rows2].clone().requires_grad_(True)

shard_b = (W1.numel() + W2.numel()) * 4
full_b = (W1_full.numel() + W2_full.numel()) * 4
if rank == 0:
    print(f"本卡参数 {shard_b / 2**20:.1f} MB（完整版 {full_b / 2**20:.1f} MB，已 ÷{world}）",
          flush=True)

comm_reset()
out = fsdp_matmul(F.gelu(fsdp_matmul(X, W1)), W2)
loss = ((out - target) ** 2).mean()
comm_report("forward")
comm_reset()                         # 清零：让 backward 的通信单独统计
loss.backward()
comm_report("backward")

ok_loss = same(loss.detach(), loss_ref.detach())
ok_w1 = same(W1.grad, g1[rank * rows1:(rank + 1) * rows1])
ok_w2 = same(W2.grad, g2[rank * rows2:(rank + 1) * rows2])
status = "OK" if (ok_loss and ok_w1 and ok_w2) else "FAIL"
print(f"rank {rank}: loss 对比 {ok_loss}，dW1 分片对比 {ok_w1}，dW2 分片对比 {ok_w2} "
      f"[{status}]", flush=True)
dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=2 exp11_fsdp.py
```

预期输出：

```text
本卡参数 16.0 MB（完整版 32.0 MB，已 ÷2）
[通信账] forward：all_gather×2（16.0 MiB）
[通信账] backward：all_gather×2（16.0 MiB）  reduce_scatter×2（16.0 MiB）
rank 0: loss 对比 True，dW1 分片对比 True，dW2 分片对比 True [OK]
rank 1: loss 对比 True，dW1 分片对比 True，dW2 分片对比 True [OK]
```

这份输出逐行看：

1. **第一行是显存**：每张卡只存 16 MB 参数，是完整版的一半——ZeRO-3 的 ÷t 直接可见。注意一个细节：这次两卡用的是**不同数据**（`manual_seed(1000 + rank)`），参考梯度是两张卡各自梯度的**和**——这正是 DP 的求和语义，由 reduce-scatter 完成。
2. **第二、三行是通信量**：前向 all-gather ×2（拼两次参数），反向 all-gather ×2（重拼）+ reduce-scatter ×2（梯度求和并取本卡分片）。三种原语都出现了——第 2 篇的原语表、第 6 篇的对偶关系，都体现在这几行里。
3. **三个 True**：loss 一致说明前向拼参数正确；梯度分片一致说明 reduce-scatter 的"求和 + 切片"结果精确。整个过程中没有长期保存全量参数，数学结果却和单卡等价。

把本篇和第 6 篇放在一起，三种并行方式的通信特点就齐了：

|  | TP（篇 6） | DP（本篇 §一~三） | FSDP（本篇 §四~七） |
| --- | --- | --- | --- |
| 通信对象 | 激活（每层 4 次） | 梯度（每 step 一次） | 参数 + 梯度（每层 3 次） |
| 单条消息 | KB~MB，延迟敏感 | GB 级，带宽敏感 | 参数片/梯度片，带宽敏感 |
| 能否合并/重叠 | 由层内依赖决定，无法重叠 | 分桶 + 异步 + 重叠，与 backward 并行 | 按层发生，prefetch 与计算并行 |
| 解决什么问题 | 单层算不下 → 切开 | —— | 每卡显存 ÷t |
| 适合部署在 | 机内（NVLink） | 跨机（IB/RoCE） | 跨机 |

## 七、实验三：这笔交换划不划算（exp12）

3P 对 2P 是纸面账。这一节回答最实际的问题：**多付的 50% 通信，到底会不会拉长训练时间？**

同一个 6 层 MLP（尺寸与 exp10 相同）、同一份初始权重、同样的数据，四组对拍：

| 组 | 常驻参数 | 通信怎么做 |
| --- | --- | --- |
| `dp_naive` | 全量 24 MB | backward 全部算完，再逐层阻塞 all-reduce |
| `dp_overlap` | 全量 24 MB | 逐层梯度就绪即异步 all-reduce（exp10 的做法） |
| `fsdp_naive` | 分片 12 MB | 每层阻塞 all-gather 拼参数（前向、反向各一次）+ 阻塞 reduce-scatter |
| `fsdp_overlap` | 分片 12 MB | 算第 i 层时预取下一层参数，梯度规约异步发起（真实 FSDP 的 prefetch 做法） |

新建 `exp12_dp_vs_fsdp.py`（自包含，不依赖 `mini_tp.py`，计数器打印后自动清零）：

```python
# exp12_dp_vs_fsdp.py —— 实验三：纯 DP vs FSDP——用通信换显存，训练时间变多少？
# 同一个 6 层 MLP、同一份初始权重、同样的数据，四组对拍：
#   DP   朴素：backward 全部算完，再逐层阻塞 all_reduce（通信整块暴露）
#   DP   重叠：逐层梯度一算出来就 async all_reduce（exp10 的做法）
#   FSDP 朴素：每层阻塞 all-gather 拼参数 + 阻塞 reduce-scatter 留切片
#   FSDP 重叠：前向/反向预取下一层参数，梯度规约异步发起（真实 FSDP 的做法）
# 看两组数字：通信量（计数器，恒定的 1.5 倍）与每 step 耗时（计时，因机器而异）。
import time
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

# ---- 通信计数器：按 op 记次数与本地缓冲字节数，打印后清零 ----
_CNT = {}
def _rec(op, t):
    _CNT.setdefault(op, [0, 0])
    _CNT[op][0] += 1
    _CNT[op][1] += t.numel() * t.element_size()
def comm_reset():
    _CNT.clear()
def comm_report(tag):
    parts = "  ".join(f"{k}×{v[0]}（{v[1]/2**20:.1f} MiB）" for k, v in sorted(_CNT.items()))
    print(f"[通信账] {tag}：{parts}", flush=True)
    _CNT.clear()

L, H, T = 6, 1024, 256          # 与 exp10 同尺寸，方便对照
lr = 0.01
torch.manual_seed(1000 + rank)  # 两卡不同数据 —— 真正的 DP
X = torch.randn(T, H)
Ytarget = torch.randn(T, H)

def gelu_prime(z):
    # gelu(x) = x·Φ(x)；导数 = Φ(x) + x·φ(x)
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def make_full():
    torch.manual_seed(42)       # 四组实验从同一份权重出发
    return [torch.randn(H, H, device="cpu") / H**0.5 for _ in range(L)]

def shard(W):                   # 沿行切，返回本卡分片（FSDP 的常驻参数）
    r = H // world
    return W[rank * r:(rank + 1) * r].clone()

def gather_async(sw):           # 异步 all-gather：投出就去干别的，句柄稍等
    outs = [torch.empty_like(sw) for _ in range(world)]
    h = dist.all_gather(outs, sw.contiguous(), async_op=True)
    return outs, h

def gather(sw):                 # 阻塞 all-gather：拼回完整参数
    outs = [torch.empty_like(sw) for _ in range(world)]
    _rec("all_gather", sw)
    dist.all_gather(outs, sw.contiguous())
    return torch.cat(outs, dim=0)

def forward(Ws):
    A = X
    As, Zs = [A], []
    for i in range(L):
        z = A @ Ws[i]
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    return As, Zs

def dA_loss(As):
    return 2 * (As[-1] - Ytarget) / As[-1].numel()   # MSE 的导数

# ---------- 纯 DP：每卡常驻全量参数，梯度 all_reduce 求和 ----------

def dp_naive(Ws):
    """朴素版：整个反向算完，再逐层阻塞 all_reduce —— 通信整块暴露。"""
    As, Zs = forward(Ws)
    dA = dA_loss(As)
    grads = [None] * L
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = As[i].T @ dz
        dA = dz @ Ws[i].T
    for i in range(L):
        _rec("all_reduce", grads[i])
        dist.all_reduce(grads[i])            # 阻塞：发起并等完
    for i in range(L):
        Ws[i] -= lr * grads[i]

def dp_overlap(Ws):
    """重叠版：每层的梯度一算出来就立刻异步 all_reduce，回头继续算上一层。"""
    As, Zs = forward(Ws)
    dA = dA_loss(As)
    grads = [None] * L
    hs = []
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        grads[i] = As[i].T @ dz
        _rec("all_reduce", grads[i])
        hs.append(dist.all_reduce(grads[i], async_op=True))  # 投单就走
        dA = dz @ Ws[i].T                    # 通信在身后飞
    for h in hs:
        h.wait()
    for i in range(L):
        Ws[i] -= lr * grads[i]

# ---------- FSDP：每卡常驻 1/t 参数分片，用到拼全、梯度求和留切片 ----------

def fsdp_naive(SWs):
    """朴素版：每层阻塞 all-gather 拼参数（前向一次、反向一次），阻塞 reduce-scatter。"""
    Wf = [gather(s) for s in SWs]            # 前向：逐层阻塞拼全
    As, Zs = forward(Wf)
    dA = dA_loss(As)
    gsh = [None] * L
    for i in reversed(range(L)):
        w = gather(SWs[i])                   # 反向：前向没留全量，再拼一次
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz                    # 本卡数据的完整梯度
        out = torch.empty_like(SWs[i])
        _rec("reduce_scatter", out)
        dist.reduce_scatter_tensor(out, gw.contiguous())   # 求和 + 只留本卡那片
        gsh[i] = out
        dA = dz @ w.T
    for i in range(L):
        SWs[i] -= lr * gsh[i]

def fsdp_overlap(SWs):
    """重叠版：算第 i 层时，第 i+1 层（反向则是第 i-1 层）的参数已在路上；
    梯度 reduce-scatter 异步发起，继续算上一层。"""
    outs, h = gather_async(SWs[0])
    _rec("all_gather", SWs[0])
    A = X
    As, Zs = [A], []
    for i in range(L):                       # 前向：预取下一层参数
        h.wait()
        w_full = torch.cat(outs, dim=0)
        if i + 1 < L:
            outs, h = gather_async(SWs[i + 1])
            _rec("all_gather", SWs[i + 1])
        z = A @ w_full
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    dA = dA_loss(As)
    outs, h = gather_async(SWs[L - 1])
    _rec("all_gather", SWs[L - 1])
    gsh = [None] * L
    rhs = []
    for i in reversed(range(L)):             # 反向：预取上一层参数
        h.wait()
        w_full = torch.cat(outs, dim=0)
        if i - 1 >= 0:
            outs, h = gather_async(SWs[i - 1])
            _rec("all_gather", SWs[i - 1])
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz
        out = torch.empty_like(SWs[i])
        _rec("reduce_scatter", out)
        rhs.append(dist.reduce_scatter_tensor(out, gw.contiguous(), async_op=True))
        gsh[i] = out
        dA = dz @ w_full.T
    for h_ in rhs:
        h_.wait()
    for i in range(L):
        SWs[i] -= lr * gsh[i]

def bench(name, fn, mk, n=5):
    """预热 1 步（不计时）+ 计时 n 步；计数器清零后只统计这 n 步。"""
    W = mk()
    fn(W)                                    # 预热：建组等成本花在计时外
    comm_reset()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(W)
        ts.append((time.perf_counter() - t0) * 1e3)
    comm_report(f"{name}（{n} 步合计）")
    print(f"{name:<13} 每 step：{'  '.join(f'{t:.1f}' for t in ts)} ms", flush=True)
    return W

if rank == 0:
    dp_mb = L * H * H * 4 / 2**20
    print(f"常驻参数：DP 每卡 {dp_mb:.0f} MB，FSDP 每卡 {dp_mb // world:.0f} MB（已 ÷{world}）", flush=True)

bench("dp_naive", dp_naive, make_full)
bench("dp_overlap", dp_overlap, make_full)
bench("fsdp_naive", fsdp_naive, lambda: [shard(w) for w in make_full()])
bench("fsdp_overlap", fsdp_overlap, lambda: [shard(w) for w in make_full()])

# 正确性：从同一份初始权重出发，DP 与 FSDP 各走 1 步，拼回分片对比
Ws = make_full()
SWs = [shard(w) for w in make_full()]
dp_overlap(Ws)
fsdp_overlap(SWs)
ok = True
for i in range(L):
    outs = [torch.empty_like(SWs[i]) for _ in range(world)]
    dist.all_gather(outs, SWs[i].contiguous())
    ok = ok and torch.allclose(torch.cat(outs, dim=0), Ws[i], rtol=1e-4, atol=1e-5)
if rank == 0:
    print(f"DP 与 FSDP 各走 1 步后权重一致：{ok}", flush=True)

dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=2 exp12_dp_vs_fsdp.py
```

预期输出（通信量是重点；以下计时来自一台 2 核 CPU + gloo 的机器，以你的实测为准）：

```text
常驻参数：DP 每卡 24 MB，FSDP 每卡 12 MB（已 ÷2）
[通信账] dp_naive（5 步合计）：all_reduce×30（120.0 MiB）
dp_naive      每 step：299.1  308.9  290.9  305.7  294.5 ms
[通信账] dp_overlap（5 步合计）：all_reduce×30（120.0 MiB）
dp_overlap    每 step：278.2  308.8  306.5  306.1  306.6 ms
[通信账] fsdp_naive（5 步合计）：all_gather×60（120.0 MiB）  reduce_scatter×30（60.0 MiB）
fsdp_naive    每 step：378.9  376.7  369.2  365.0  369.1 ms
[通信账] fsdp_overlap（5 步合计）：all_gather×60（120.0 MiB）  reduce_scatter×30（60.0 MiB）
fsdp_overlap  每 step：391.3  390.4  360.0  381.5  386.0 ms
DP 与 FSDP 各走 1 步后权重一致：True
```

四组数，四个结论：

1. **通信量精确符合纸面账**：DP 每 step 24 MiB（6 层 × 4 MB 梯度）、FSDP 36 MiB（拼参数两份 + 梯度一份）——36/24 = 1.5，3P/2P 的实测版。
2. **朴素版确实更慢**：`fsdp_naive` 比 `dp_naive` 慢约 25%，慢出来的时间约等于多走 12 MiB 通信的时间。带宽换显存不是免费的，标价就写在这。
3. **重叠是还价手段**：这台 2 核机器上 `fsdp_overlap` 没有追上——核太少，通信线程和计算线程在抢同一批核，重叠物理上施展不开（附录 A 第一条说过这个现象）。在核数足够的机器上，`dp_overlap` 会明显快于 `dp_naive`，`fsdp_overlap` 随之追上 `dp_overlap`。
4. **GPU 上差距进一步缩小**：NCCL 通信走独立 stream、传输走 DMA 引擎，多付的通信大部分能被藏进计算。这就是工程上"FSDP 几乎总是划算"的底气：1.5 倍通信可以重叠，显存 ÷t 却是装不下就训练不能开始。

另外注意输出的第一行：同一份模型，FSDP 每卡常驻参数是 DP 的一半（t=2）；换成 t=64、175B 模型的场合，这一行就是 2.8 TB 对 44 GB。

## 八、如果这 50% 通信接受不了：ZeRO-1 与 ZeRO-2

§七 的账算完了：FSDP 用 1.5 倍通信换 ÷t 的常驻显存，撞上显存墙时几乎总是划算。但"几乎总是"不是"总是"——两种情况这笔交换是亏的：一是**模型没大到那份上**，7B 用 8 卡纯 DP 要 112 GB、差一口气，其实用不着一路切到 16/t；二是**带宽本来就紧**，跨机网络跑满之后，多付的 50% 通信会 1:1 变成训练时间的拉长。这时该回头把 §四 没展开的分档表摆全了——FSDP 只是 ZeRO 三档里切得最狠的那档，前两档的通信账完全不同：

| 档位 | 切什么 | 每卡每参数 | t=8、7B 模型每卡 |
| --- | --- | --- | --- |
| 纯 DP | 不切 | 16 B | 112 GB |
| ZeRO-1 | 优化器状态 ÷t | 2+2+4+8/t B | 63 GB |
| ZeRO-2 | 再切梯度 ÷t | 2+4+(2+8)/t B | 51 GB |
| ZeRO-3（= FSDP） | 连参数一起切 ÷t | **16/t B** | **14 GB** |

通信量按本篇的计费标准（all-gather / reduce-scatter 搬一份消息记 P，all-reduce 记 2P，忽略 (t-1)/t 系数）：

| 档位 | 每 step 通信 | 花在哪 |
| --- | --- | --- |
| 纯 DP | 2P | 梯度 all-reduce：求和 P + 结果发回全组 P |
| ZeRO-1 | 2P | 梯度 reduce-scatter 求和留切片 P + 更新后的参数分片 all-gather 发回全组 P |
| ZeRO-2 | 2P | 与 ZeRO-1 一字不差 |
| ZeRO-3（= FSDP） | 3P | 参数前向拼全 P + 反向重拼 P + 梯度 reduce-scatter P |

三层意思：

1. **ZeRO-1 是纯赚**。通信凭什么还是 2P？注意梯度规约从 all-reduce 换成了 reduce-scatter：每卡本来只需要"自己要更新的那 1/t 参数"对应的梯度，all-reduce 把结果发回全组的那一半（P）纯属浪费——省下来正好抵掉更新后参数 all-gather 的那 P。m、v 只是优化器的内部状态，切它不影响前向反向的任何计算；常驻从 16 降到 2+2+4+8/t。只要用 Adam，这一档没有任何理由不开。
2. **ZeRO-2 通信不变，再切梯度**。它和 ZeRO-1 的通信一模一样（同为 RS + AG），差别只在显存：ZeRO-1 的全量梯度要攒到反向结束才规约，ZeRO-2 把 reduce-scatter 挪进反向逐层发起，每层梯度算完即规约、只留分片——全量梯度从"常驻"降级为"单层瞬时"，常驻再降到 2+4+(2+8)/t。这个手法眼熟吗？就是 §二 exp10 里 DP 从朴素到重叠的那一手，只不过这一刀省下的不（只）是时间，是显存。
3. **ZeRO-3 才开始真正多付**。常驻参数没了，"发回全组"从梯度换成参数还不够——参数得用前拼、用后扔：前向拼一次、反向重拼一次共 2P，加上梯度规约 P，合计 3P。它比纯 DP 只多 1P：梯度求和后不再发回全组（每卡只更新自己的分片），那一半永远省下了。净增 50%，换常驻 ÷t——这正是 §五 到 §七 已经验证过的那笔交换。

### 8.1 代码：ZeRO-1 怎么做——和 FSDP 的完整流程对照

先回答一个前置问题：ZeRO-1 和 FSDP 是同一个功能吗？**是**。两者实现的是完全相同的训练 step——前向算 loss、反向算梯度、梯度跨卡求和、Adam 更新——数学语义一模一样（exp13 末尾"四种方式权重一致 True"就是证明）。不同的只有一件事：**每样数据（参数、梯度、优化器状态）以什么形态存在哪张卡上、什么时候搬**。所以正确的看法不是各摘一段循环对着看，而是把完整流程按相同阶段对齐。

一个训练 step 都可以拆成五段：①前向 ②反向 ③梯度规约 ④更新 ⑤收尾。下面是两个完整实现（为对齐结构，把 exp13 的 `step_fsdp` 去掉预取、改成阻塞拼全，逻辑等价）：

```python
# ———— ZeRO-1 的完整 step ————
def step_zero1(W, m, v):            # W 全量常驻；m、v 只有本卡负责的 1/t
    # ① 前向：全量参数就在手边，直接算 —— 零通信
    As, Zs = forward(W)
    # ② 反向：还是直接算 —— 零通信
    dA = dA_loss(As)
    G = [None] * L
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        G[i] = As[i].T @ dz
        dA = dz @ W[i].T
    # ③④⑤ 反向后逐层：规约 → 更新分片 → 拼回全量
    for i in range(L):
        gsh = torch.empty(r, H)
        dist.reduce_scatter_tensor(gsh, G[i].contiguous())   # ③ 梯度求和，只留本卡分片
        adam_(W[i][rank*r:(rank+1)*r], gsh, m[i], v[i])      # ④ 只更新自己那 1/t
        outs = [torch.empty(r, H) for _ in range(world)]
        dist.all_gather(outs, W[i][rank*r:(rank+1)*r])       # ⑤ 把更新后的分片拼回全量
        W[i].copy_(torch.cat(outs, dim=0))
```

```python
# ———— FSDP 的完整 step ————
def step_fsdp(SW, m, v):            # SW 只有 1/t 分片常驻；m、v 也只有 1/t
    # ① 前向：每层先 all_gather 拼全，算完即释放
    A = X
    As, Zs = [A], []
    for i in range(L):
        outs = [torch.empty_like(SW[i]) for _ in range(world)]
        dist.all_gather(outs, SW[i].contiguous())      # ← 通信发生在前向里
        w_full = torch.cat(outs, dim=0)
        z = A @ w_full
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    # ②③ 反向逐层：重拼参数 → 算梯度 → 规约留片
    dA = dA_loss(As)
    gsh = [None] * L
    for i in reversed(range(L)):
        outs = [torch.empty_like(SW[i]) for _ in range(world)]
        dist.all_gather(outs, SW[i].contiguous())      # ← 前向没留全量，只能再拼一次
        w_full = torch.cat(outs, dim=0)
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz
        gsh[i] = torch.empty_like(SW[i])
        dist.reduce_scatter_tensor(gsh[i], gw.contiguous())
        dA = dz @ w_full.T
    # ④ 更新：只更新自己的分片 —— 分片就是全部，没有⑤
    for i in range(L):
        adam_(SW[i], gsh[i], m[i], v[i])
```

五段对照：

| 阶段 | ZeRO-1 | FSDP |
| --- | --- | --- |
| ① 前向 | 全量参数直接算，**零通信** | 每层 all_gather 拼全参数，算完即释放 |
| ② 反向 | 全量参数直接算，**零通信** | 每层**再** all_gather 一次（前向没留全量） |
| ③ 梯度规约 | reduce_scatter 求和留片（反向后统一做） | reduce_scatter 求和留片（反向中逐层做） |
| ④ 更新 | 只更新自己的 1/t 分片 | **完全相同** |
| ⑤ 收尾 | all_gather 把更新后的分片拼回全量 | **没有这一阶段**——分片即终态 |

对着表看，结论就清楚了：

- **③④ 两段两者完全相同**——都是"求和留片 + 只更新分片"，代码行几乎一字不差。两档的差异不在梯度端，全在**参数端**。
- **ZeRO-1 的通信全部在计算之外**：③⑤ 两次（RS P + AG P = 2P），前向反向零通信。**FSDP 的通信全部在计算之中**：①②③ 三次（AG + AG + RS = 3P）——这就是为什么 FSDP 多付的那 1P 能按层 prefetch 重叠（§五），而 ZeRO-1 的两次通信只能硬等。ZeRO-2 把 ③ 挪进反向，正是为了把其中一次也变成可重叠，这是下一小节的内容。
- **⑤ 是 ZeRO-1 特有的**：每卡只更新了 1/t，下一个 step 的前向却要用全量参数，不拼回来各卡参数就分叉。FSDP 不需要⑤——它的"全量参数"从不存在，每次用到都现拼，代价因此记在①②头上。
- 一句话总结：**ZeRO-1 用"更新后拼回一次"换"计算时零通信"；FSDP 用"计算时拼两次"换"常驻显存 ÷t"**。

### 8.2 代码：ZeRO-2 = ZeRO-1 + 一手

ZeRO-1 的全量梯度要攒到反向结束（`G` 列表装满 6 层才开始规约），ZeRO-2 把规约挪回反向循环里，一层一清：

```python
# ZeRO-2 的反向，每层：
gw = As[i].T @ dz                              # 本层全量梯度，循环体结束即释放
hs.append(dist.reduce_scatter_tensor(gsh[i], gw, async_op=True))  # 规约投单就走
dA = dz @ W[i].T                               # 不停顿，继续算上一层
# 反向结束后：统一 wait → 更新分片 → all_gather 参数，与 ZeRO-1 相同
```

对照 ZeRO-1 的那段循环，改动就是 §二 的三板斧再来一遍：规约改 `async_op=True`、挪进反向循环、wait 延后到反向结束。通信字节一个不少（仍是 RS P + AG P = 2P），但两个变化实打实：全量梯度不再常驻（每层算完即释放，常驻只剩 1/t 分片——这就是"梯度 ÷t"的代码形态），规约还能与反向计算重叠。可以说 **ZeRO-2 之于 ZeRO-1，就是重叠版 DP 之于朴素版 DP**——只是这次省下的主要是显存。

三档的代码差异收拢成一张表（AG = all-gather，RS = reduce-scatter）：

|  | 反向中的通信 | 反向结束后的通信 | 每卡常驻 | 每 step 通信 |
| --- | --- | --- | --- | --- |
| ZeRO-1 | 无 | 梯度 RS + 参数 AG | 参数全量、梯度全量（攒到反向结束）、优化器 ÷t | 2P |
| ZeRO-2 | 梯度 RS（逐层异步） | 参数 AG | 参数全量、梯度 ÷t、优化器 ÷t | 2P |
| ZeRO-3（FSDP） | 参数 AG×2 + 梯度 RS | 无 | 全部 ÷t | 3P |

### 8.3 实验四：三档与 DP 对拍——显存、通信、时间（exp13）

把三档和纯 DP 放进同一个脚本对拍。四组共用同一个 6 层 MLP（尺寸与 exp10/12 相同）、同一份初始权重、同样的数据，以及**同一条 mini-Adam 更新规则**——优化器状态 `m`、`v` 这次是真实分配、真实参与更新的；"显存账"不再是纸面推算，而是反向结束时把本卡实际持有的张量字节加总打印。新建 `exp13_zero_stages.py`：

```python
# exp13_zero_stages.py —— 实验四：ZeRO 三档与 DP 对拍——显存、通信量、时间各是多少？
# 同一个 6 层 MLP、同一份初始权重、同样的数据、同一条 mini-Adam 更新规则，四组对拍：
#   dp    ：全量参数 + 全量梯度 + 全量优化器状态；梯度逐层异步 all_reduce（exp10 的做法）
#   zero1 ：参数全量常驻，优化器状态 ÷t；反向零通信，反向后梯度 reduce_scatter 留片，
#           各卡只更新自己的分片，再 all_gather 把参数拼回全量（通信与 DP 同为 2P）
#   zero2 ：在 zero1 基础上把梯度 reduce_scatter 挪进反向逐层异步发起——
#           全量梯度从"攒到反向结束"变成"单层瞬时"（通信不变，梯度显存 ÷t）
#   fsdp  ：参数也 ÷t，用时 all_gather 拼全、用完即释放（exp12 的做法）
# 看三组数字：反向结束时本卡实际持有字节、通信量（计数器）、每 step 耗时。
import time
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 2, "请用 torchrun --nproc_per_node=2 运行"

# ---- 通信计数器：按 op 记次数与本地缓冲字节数，打印后清零 ----
_CNT = {}
def _rec(op, t):
    _CNT.setdefault(op, [0, 0])
    _CNT[op][0] += 1
    _CNT[op][1] += t.numel() * t.element_size()
def comm_reset():
    _CNT.clear()
def comm_report(tag):
    parts = "  ".join(f"{k}×{v[0]}（{v[1]/2**20:.1f} MiB）" for k, v in sorted(_CNT.items()))
    print(f"[通信账] {tag}：{parts}", flush=True)
    _CNT.clear()

L, H, T = 6, 1024, 256          # 与 exp10/12 同尺寸，方便对照
lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
torch.manual_seed(1000 + rank)  # 两卡不同数据 —— 真正的 DP
X = torch.randn(T, H)
Yt = torch.randn(T, H)

def gelu_prime(z):
    Phi = 0.5 * (1 + torch.erf(z / 2**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2 * torch.pi) ** 0.5
    return Phi + z * pdf

def make_full():
    torch.manual_seed(42)       # 四组实验从同一份权重出发
    return [torch.randn(H, H) / H**0.5 for _ in range(L)]

r = H // world                  # 每卡分片的行数
def shard(W):
    return W[rank * r:(rank + 1) * r].clone()

def forward(Ws):
    A = X
    As, Zs = [A], []
    for i in range(L):
        z = A @ Ws[i]
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    return As, Zs

def dA_loss(As):
    return 2 * (As[-1] - Yt) / As[-1].numel()

def adam_(p, g, m, v, t):       # 就地 mini-Adam：四组共用同一条更新规则，结果才能互校
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    p.add_(mh / (vh.sqrt() + eps), alpha=-lr)

# ---- 显存账：反向结束时本卡实际持有的张量（参数 / 优化器状态 / 梯度），只记一次 ----
_PEAK = []
def note_peak(*groups):
    if not _PEAK:                        # 立即折算成字节快照，后续释放不影响
        _PEAK.extend(sum(t.numel() * t.element_size() for t in g) for g in groups)
def peak_report(tag):
    names = ("参数", "优化器状态", "梯度")
    detail = " + ".join(f"{n} {b / 2**20:.0f}" for n, b in zip(names, _PEAK))
    if rank == 0:
        print(f"[显存账] {tag}：反向结束时本卡持有 {sum(_PEAK) / 2**20:.0f} MiB"
              f"（{detail} MiB）", flush=True)

def gather_async(sw):
    outs = [torch.empty_like(sw) for _ in range(world)]
    h = dist.all_gather(outs, sw.contiguous(), async_op=True)
    return outs, h

# ---------- dp：全量参数、全量梯度、全量优化器状态 ----------
def step_dp(st, t):
    W, m, v = st
    As, Zs = forward(W)
    dA = dA_loss(As)
    G, hs = [None] * L, []
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        G[i] = As[i].T @ dz
        _rec("all_reduce", G[i])
        hs.append(dist.all_reduce(G[i], async_op=True))
        dA = dz @ W[i].T
    note_peak(W, m + v, G)                       # 全量梯度一直攒到反向结束
    for h in hs:
        h.wait()
    for i in range(L):
        adam_(W[i], G[i], m[i], v[i], t)         # 更新全量参数（优化器状态也是全量）

# ---------- zero1：参数全量，优化器状态 ÷t；反向零通信 ----------
def step_zero1(st, t):
    W, m, v = st
    As, Zs = forward(W)
    dA = dA_loss(As)
    G = [None] * L
    for i in reversed(range(L)):                 # 反向：参数在手边，没有任何通信
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        G[i] = As[i].T @ dz
        dA = dz @ W[i].T
    note_peak(W, m + v, G)                       # 全量梯度攒到反向结束
    for i in range(L):
        gsh = torch.empty(r, H)
        _rec("reduce_scatter", gsh)
        dist.reduce_scatter_tensor(gsh, G[i].contiguous())  # 求和 + 只留本卡分片
        G[i] = None                              # 全量梯度用完即扔
        sl = W[i][rank * r:(rank + 1) * r]
        adam_(sl, gsh, m[i], v[i], t)            # 只更新自己负责的那 1/t
        outs = [torch.empty(r, H) for _ in range(world)]
        _rec("all_gather", outs[rank])
        dist.all_gather(outs, sl.contiguous())   # 把更新后的分片拼回全量参数
        W[i].copy_(torch.cat(outs, dim=0))

# ---------- zero2：在 zero1 基础上把梯度规约挪进反向，逐层异步 ----------
def step_zero2(st, t):
    W, m, v = st
    As, Zs = forward(W)
    dA = dA_loss(As)
    gsh, hs = [None] * L, []
    for i in reversed(range(L)):
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz                        # 本层全量梯度，循环体结束即释放
        gsh[i] = torch.empty(r, H)
        _rec("reduce_scatter", gsh[i])
        hs.append(dist.reduce_scatter_tensor(gsh[i], gw.contiguous(), async_op=True))
        dA = dz @ W[i].T                         # 通信与上一层的计算并行
    note_peak(W, m + v, gsh)                     # 梯度常驻只剩分片
    for h in hs:
        h.wait()
    for i in range(L):
        sl = W[i][rank * r:(rank + 1) * r]
        adam_(sl, gsh[i], m[i], v[i], t)
        outs = [torch.empty(r, H) for _ in range(world)]
        _rec("all_gather", outs[rank])
        dist.all_gather(outs, sl.contiguous())
        W[i].copy_(torch.cat(outs, dim=0))

# ---------- fsdp：参数也 ÷t，用时拼全、用完即释放 ----------
def step_fsdp(st, t):
    SW, m, v = st
    outs, h = gather_async(SW[0])
    _rec("all_gather", SW[0])
    A = X
    As, Zs = [A], []
    for i in range(L):                           # 前向：预取下一层参数
        h.wait()
        w = torch.cat(outs, dim=0)
        if i + 1 < L:
            outs, h = gather_async(SW[i + 1])
            _rec("all_gather", SW[i + 1])
        z = A @ w
        Zs.append(z)
        A = z if i == L - 1 else torch.nn.functional.gelu(z)
        As.append(A)
    dA = dA_loss(As)
    outs, h = gather_async(SW[L - 1])
    _rec("all_gather", SW[L - 1])
    gsh, rhs = [None] * L, []
    for i in reversed(range(L)):                 # 反向：预取上一层参数
        h.wait()
        w = torch.cat(outs, dim=0)
        if i - 1 >= 0:
            outs, h = gather_async(SW[i - 1])
            _rec("all_gather", SW[i - 1])
        dz = dA if i == L - 1 else dA * gelu_prime(Zs[i])
        gw = As[i].T @ dz
        gsh[i] = torch.empty_like(SW[i])
        _rec("reduce_scatter", gsh[i])
        rhs.append(dist.reduce_scatter_tensor(gsh[i], gw.contiguous(), async_op=True))
        dA = dz @ w.T
    note_peak(SW, m + v, gsh)                    # 参数、梯度、优化器状态全是分片
    for h_ in rhs:
        h_.wait()
    for i in range(L):
        adam_(SW[i], gsh[i], m[i], v[i], t)      # 更新分片即终态，无需拼回

def bench(tag, step_fn, mk_state, n=5):
    global _PEAK
    _PEAK = []
    st = mk_state()
    step_fn(st, 1)                               # 预热（建组等成本不计时），同时记下显存账
    peak_report(tag)
    comm_reset()
    ts = []
    for k in range(2, n + 2):
        t0 = time.perf_counter()
        step_fn(st, k)
        ts.append((time.perf_counter() - t0) * 1e3)
    if rank == 0:
        comm_report(f"{tag}（{n} 步合计）")
        print(f"{tag:<7} 每 step：{'  '.join(f'{x:.1f}' for x in ts)} ms", flush=True)

def mk_dp():    return make_full(), [torch.zeros(H, H) for _ in range(L)], [torch.zeros(H, H) for _ in range(L)]
def mk_z1():    return make_full(), [torch.zeros(r, H) for _ in range(L)], [torch.zeros(r, H) for _ in range(L)]
def mk_fsdp():  return [shard(w) for w in make_full()], [torch.zeros(r, H) for _ in range(L)], [torch.zeros(r, H) for _ in range(L)]

bench("dp",     step_dp,    mk_dp)
bench("zero1",  step_zero1, mk_z1)
bench("zero2",  step_zero2, mk_z1)
bench("fsdp",   step_fsdp,  mk_fsdp)

# 正确性：四组从同一份初始权重出发各走 1 步，FSDP 拼回分片，互相校验
st_dp, st_z1, st_z2, st_f = mk_dp(), mk_z1(), mk_z1(), mk_fsdp()
step_dp(st_dp, 1); step_zero1(st_z1, 1); step_zero2(st_z2, 1); step_fsdp(st_f, 1)
ok = True
for i in range(L):
    outs = [torch.empty_like(st_f[0][i]) for _ in range(world)]
    dist.all_gather(outs, st_f[0][i].contiguous())
    wf = torch.cat(outs, dim=0)
    ok = ok and torch.allclose(wf, st_dp[0][i], rtol=1e-4, atol=1e-5) \
             and torch.allclose(wf, st_z1[0][i], rtol=1e-4, atol=1e-5) \
             and torch.allclose(wf, st_z2[0][i], rtol=1e-4, atol=1e-5)
if rank == 0:
    print(f"四种方式各走 1 步后权重一致：{ok}", flush=True)

dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=2 exp13_zero_stages.py
```

预期输出（显存账与通信账是重点；以下计时来自一台 2 核 CPU + gloo 的机器，以你的实测为准）：

```text
[显存账] dp：反向结束时本卡持有 96 MiB（参数 24 + 优化器状态 48 + 梯度 24 MiB）
[通信账] dp（5 步合计）：all_reduce×30（120.0 MiB）
dp      每 step：384.1  394.3  708.9  485.8  734.3 ms
[显存账] zero1：反向结束时本卡持有 72 MiB（参数 24 + 优化器状态 24 + 梯度 24 MiB）
[通信账] zero1（5 步合计）：all_gather×30（60.0 MiB）  reduce_scatter×30（60.0 MiB）
zero1   每 step：371.8  367.8  720.4  488.0  647.2 ms
[显存账] zero2：反向结束时本卡持有 60 MiB（参数 24 + 优化器状态 24 + 梯度 12 MiB）
[通信账] zero2（5 步合计）：all_gather×30（60.0 MiB）  reduce_scatter×30（60.0 MiB）
zero2   每 step：368.8  374.0  723.8  502.3  667.9 ms
[显存账] fsdp：反向结束时本卡持有 48 MiB（参数 12 + 优化器状态 24 + 梯度 12 MiB）
[通信账] fsdp（5 步合计）：all_gather×60（120.0 MiB）  reduce_scatter×30（60.0 MiB）
fsdp    每 step：417.0  373.8  770.9  541.4  704.2 ms
四种方式各走 1 步后权重一致：True
```

（绝对耗时比 exp12 略高：Adam 更新、参数拼回的拷贝也计在 step 内；看组间相对差即可。）

三组数字正好是分档表的实测版：

1. **显存账逐级下降，比例精确**：96 → 72 → 60 → 48 MiB。脚本是 fp32 全程：参数 4 + 梯度 4 + m 4 + v 4 = 16 B/param——总额碰巧与 §四 混合精度口径的 16 B 相同，分档公式换成 fp32 版就是 DP 16、ZeRO-1 8+8/t、ZeRO-2 4+12/t、ZeRO-3 16/t，t=2 乘上 6M 参数，正好就是这四个数。注意 ZeRO-1 → ZeRO-2 省下的 12 MiB 全部来自梯度（24 → 12），而通信账一个字节没变——§8.2 那句"通信不变，再切梯度"被直接打印出来了。
2. **通信账只有两档**：dp / zero1 / zero2 同为每 step 24 MiB（120 ÷ 5），fsdp 36 MiB（180 ÷ 5）——纸面的 2P / 2P / 2P / 3P 一字不差。ZeRO-1 多出来的 all_gather ×30，就是它"更新后把参数拼回全组"的那 P。
3. **时间差在噪声量级，方向别太当真**：四组耗时贴在同一区间。在 2 核机器上可能看到 fsdp 略慢（通信线程与计算线程抢核，zero2 的重叠收益也施展不开，与 exp12 附录 A 是同一个现象）；在核数更多的机器上甚至可能看到 **fsdp 略快**——这也不矛盾：单机 gloo 走 loopback，带宽几 GB/s 起步，fsdp 每 step 多付的 12 MiB 折成时间只有几毫秒，占比不足 1%，通信差根本浮不出来；而四组的计算量完全相同，时间差主要来自计算之外的**内存搬运**——更新环节 dp 要读写全量的 p/g/m/v（每 step 约 200 MB 流量），zero1/2 减半但多了拼回全量参数的 `cat`+`copy_`，fsdp 全是分片、流量最小。通信费说 fsdp 慢，搬运费说 fsdp 快，单机上后者赢。**3P vs 2P 的时间代价要到通信真正贵的地方才显现**：两台机器跨网络跑，或 GPU + NCCL 的机器（exp12 结论 4 的场景）。所以本节时间只作参考，硬指标是上面的显存账与通信账。
4. **权重一致 True**：四组从同一份初始权重出发、用同一条 Adam 规则各走一步，结果互相吻合——切分只是存储与通信的组织方式，数学语义始终是同一个 DP。

选型最后就一句话：**先看显存墙在哪**。装得下、带宽紧 → ZeRO-1/2（通信与 DP 相同，白捡显存）；装不下 → FSDP（多付 50% 可重叠的通信，换 ÷t 常驻）。t 越大 ÷t 越狠，FSDP 越值——t=8 时三档是 63 / 51 / 14 GB，差距会从"省一点"变成"能不能训练"。

## 九、收尾

回到开头的四个问题：

1. **怎么重叠**：梯度在 backward 中逐层倒序就绪，先就绪先发送。分桶（一层一桶，决定"哪些一起发"）+ 异步（`async_op=True`，把"发起"和"等待"拆开）+ 重叠（发起后不等待，继续算上一层）——三个手段把 DP 每 step 必须做的梯度规约，从"backward 之后的整块暴露"变成"与 backward 并行"。通信量没有变，时间省了下来。
2. **为什么成立**：分桶依赖"梯度按层就绪"和大消息带宽利用率高（第 5 篇）；异步依赖第 3 篇的 Work 句柄；重叠依赖通信与计算使用不同资源（gloo 的通信线程、NCCL 的独立 stream）。
3. **FSDP 多付了多少、为什么也能重叠**：每 step ≈3P 对纯 DP 的 ≈2P，多 50% 通信，换回每卡显存 16 B → 16/t B。多出的那一 P（参数重拼）同样按层发生，可以用 prefetch 与计算并行。
4. **接受不了这 50% 通信怎么办**（§八，exp13）：退回 ZeRO-1/2——通信与 DP 同为 2P（梯度 reduce-scatter P + 更新后参数 all-gather P），代码上只是把 FSDP 反向里的参数 all-gather 删掉、更新后补一次拼回；ZeRO-2 再把梯度规约挪进反向逐层异步，通信不变、梯度显存 ÷t。exp13 实测四档：显存 96 → 72 → 60 → 48 MiB（t=2），通信 24 / 24 / 24 / 36 MiB，时间在单机 loopback 上拉不开差距（多付的 12 MiB/step 折成时间不足 1%）。选型看显存墙的位置：装得下、带宽紧用 ZeRO-1/2，装不下才上 FSDP；t 越大 ÷t 越狠，FSDP 越值。

到这里，并行策略层的三种主要方式（TP / DP / FSDP）都手写验证过了，而且能看到一条贯穿的线索——**all-reduce = reduce-scatter + all-gather** 这个等式，在 SP 拆了一次、在 FSDP 又拆了一次；第 2 篇的原语表，覆盖的场景越来越多。

**练习题**（按难度排序，欢迎把结果贴在评论区）：

1. 把 `exp10` 的 assert 改成 `world == 4` 用 4 进程跑：通信量（次数、每卡字节数）变吗？时间呢？想想 ring all-reduce 的每卡字节系数 `2(t-1)/t` 在统计里体现在哪、没体现在哪。
2. 把 `step_overlapped` 里的 `wait` 循环挪进反向循环（每发一桶立刻 wait）：时间会退化成什么样？和 naive 的差距还剩多少？——亲手验证"省时间的是'发起后不等待'，不是 async 本身"。
3. （GPU）`TRACE=1` 跑 `exp10`，在 chrome://tracing 里找到那 6 个 all_reduce 的 kernel：各自开始的时刻对应哪一层的 backward？再仿照写一个 TRACE 分支包着 `step_naive`，把两条时间线截图对比——这就是"通信重叠"的直观体现。
4. 把 `exp11` 的 assert 改成 `world == 4`：本卡参数、通信量各变成多少？参考梯度循环已按 `world` 写好，对比还能通过吗？
5. 思考题：把 `w_full` 存进 `ctx.save_for_backward`、反向直接用——代码更省事了，显存退回哪一档？用第一行的打印算一算两种写法每卡参数各占多少。
6. 思考题：真实 DDP 构造时会先把 rank 0 的权重 broadcast 给全组（第 2 篇练习题 2 的做法），之后各卡才各自训练。为什么不能"各卡同种子、各自初始化"？提示：对比本篇"同种子造完整权重"的实验和真实训练的差别——随机初始化除了权重还消耗了什么随机状态？
7. 思考题：把 `exp13` 里 ZeRO-1 的 `reduce_scatter_tensor` 换回 `all_reduce`（每卡拿全量梯度、只更新自己的分片），语义仍然正确，但通信量从 2P 变成多少？——这解释了 ZeRO 为什么用 reduce-scatter 做梯度规约；顺便再看一眼：all-reduce = reduce-scatter + all-gather，这个等式在本篇已经是第三次出场。
8. 把 `exp13` 的 assert 改成 `world == 4` 重跑：四组的显存账、通信账各变成多少？通信量随 t 涨吗，显存呢？——亲手验证"t 越大，FSDP 越值"。

---

## 附录 A：常见问题

| 症状 | 原因与处置 |
| --- | --- |
| overlapped 不比 naive 快，甚至更慢 | gloo/CPU 下通信线程和计算线程竞争 CPU 核，属正常；GPU 机器上 NCCL 走独立 stream，效果明显。也可调大 `H`/`T` 让通信占比变高再看 |
| 两种方式权重对比 False | async 句柄没 wait 就 update，读到的是未规约梯度；或 handles 的顺序和 grads 对不上 |
| 计时第一次特别慢 | 懒初始化建组（第 3 篇介绍过），`time_it`/`bench` 里已含一次预热 |
| `exp12` 四组计时几乎一样或抖动大 | CPU 核数太少（通信线程与计算线程抢核）或通信占比太低；调大 `H`/`L` 提高通信占比，或换核数更多的机器 / GPU 复测 |
| `exp11` 对比 FAIL | 查种子顺序：先 `manual_seed(0)` 造完整权重、再 `manual_seed(1000+rank)` 造数据，且参考循环里 randn 的调用次序必须与各卡一致；查切分维度（沿行切） |
| `exp11` 里 backward 的 all_gather 显示 ×4 而不是 ×2 | 你那份 `mini_tp.py` 的 `comm_report` 只打印不清零，backward 行是 forward+backward 的累计值；代码中 forward/backward 之间已加 `comm_reset()`，加一行即可对齐 |
| `exp13` 四组计时几乎一样或抖动大 | 与 exp12 同因：CPU 核少、通信占比低；调大 `H`/`L` 或换 GPU 复测。显存账与通信账不受机器影响，任何环境都应精确复现 96/72/60/48 MiB 与每 step 24/24/24/36 MiB |
| `exp13` 里 fsdp 比 dp 还快 | 正常：单机 loopback 带宽太高，多付的 12 MiB/step 折成时间不足 1%；四组计算量相同，时间差由更新环节的内存搬运量决定（fsdp 的分片更新流量最小、无拼回拷贝）。想看 3P vs 2P 的时间差，用两台机器跨网络跑，或调小 `T`、调大 `H` 提高通信占比 |
| `exp13` 权重对比 False | 查分片起点 `rank*r` 各处是否一致；查四组 Adam 的步数 `t` 是否同步；查 ZeRO-1/2 的 `all_gather` 是否发生在更新**之后** |
| `reduce_scatter_tensor` 报 shape 错 | input 的第 0 维必须是 output 的 world 倍——沿行切参数后这是自然成立的，报错多半是张量方向弄反了 |
| 太慢或 OOM | 调小 `H`、`T`、`L`；本实验逻辑与通信结构不变 |

## 附录 B：本篇文件清单

```text
数据盘/tp-lab/
├── mini_tp.py              # 第二版：lab_init + 通信计数器 + mappings 六函数（本篇沿用）
├── exp10_dp_overlap.py     # 本篇：DP 梯度同步——朴素阻塞 vs 分桶异步重叠
├── exp11_fsdp.py           # 本篇：mini-FSDP——参数 all-gather、梯度 reduce-scatter，与单卡对比
├── exp12_dp_vs_fsdp.py     # 本篇：纯 DP vs FSDP 四组对拍——用通信换显存的时间代价
└── exp13_zero_stages.py    # 本篇：ZeRO 三档与 DP 对拍——显存、通信量、时间
```

全部 `torchrun --nproc_per_node=2 脚本名.py` 运行（gloo/笔记本可跑，计时数字以 GPU 实测为准）。

## 下篇预告

《流水线并行（PP）：气泡与 micro-batch 调度》——TP 无法避免通信，DP 把通信与计算重叠，FSDP 用通信换显存；PP 面对的既不是延迟也不是带宽，而是第三种问题：**空等**。模型按层切成几段、卡与卡接力，后面的卡在等前面的卡——时间线上会出现一块块空闲的"气泡"。micro-batch 调度（1F1B）怎么压缩气泡，第 2 篇的 send/recv 第一次成为主角——点对点通信里，进程组的"集合通信"帮不上忙，全靠两两直接收发。