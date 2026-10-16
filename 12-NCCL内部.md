# NCCL 内部：环、树、协议与 channel

> 连载第 12 篇 · NCCL 内部（上）
>
>
> 上一篇把 `dist.all_reduce` 在 c10d 里的十二步全部读完，停在了那句 `fn(inputs[0], outputs[0], comm, ncclStream)` 上——括号里的 `comm` 是建组时生成的 NCCL 通信子，`ncclStream` 是选好的队列，调用本身则是 `ncclAllReduce`。这一篇正式进入 NCCL：这句调用进去之后，NCCL 凭什么决定沿环还是沿树、用哪种协议、分几条 channel，最后又是怎么把这些决定变成一个 GPU kernel 的。
>
> 读法和上一篇相同，先概念后代码，每一站都配实验。本篇读的代码来自 NCCL 2.27（torch 2.8 自带的版本，`python -c "import torch; print(torch.cuda.nccl.version())"` 可查），引用时有删节，真实代码以你 checkout 的版本为准。消息大小的扫描实验会用到第 3 篇练习题 2 记录过的协议切换点，本篇给出它的成因。
>
> 本篇的范围是集合通信（以 all_reduce 为样本）。另外两个话题不在本篇展开：一是篇 8 用过的 send/recv——它在 NCCL 里走的是另一条路径；二是跨机时 proxy 线程的推进循环——第 5 篇只讲了它与其他两方的分工，没读代码。这两件事连同 NVLS、CollNet，留给下一篇。

**本篇的三个问题**

1. `ncclAllReduce` 进去之后，算法（ring/tree）、协议（LL/LL128/Simple）、channel 数分别在哪一步、依据什么被确定？
2. ring 为什么能逼近链路带宽上限？第 3 篇给出的"每卡上线字节 = 2(p-1)/p × 消息大小"是怎么推出来的？
3. LL、LL128、Simple 三种协议差在哪里？NCCL 怎么按消息大小在它们之间切换？

**动手指南**：本篇有四个实验脚本，全部是 GPU/NCCL 专属（gloo 不经过 NCCL，这些实验在 CPU 模式下没有意义）。除实验一需要设置环境变量外，运行方式照旧：

```bash
NCCL_TOPO_DUMP_FILE=/tmp/topo.xml NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  torchrun --nproc_per_node=2 exp25_topo.py        # 实验一：拓扑与环的日志
torchrun --nproc_per_node=N exp26_ring_scale.py    # 实验二：N=2/4/8 各跑一次
NCCL_PROTO=LL torchrun --nproc_per_node=2 exp27_proto_sweep.py   # 实验三：协议扫描
NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 \
  torchrun --nproc_per_node=8 exp28_channels.py    # 实验四：channel 数扫描
```

---

## 一、进门之后：一次 ncclAllReduce 在 NCCL 里走哪几步

先看入口本身。`ncclAllReduce` 的函数体出乎意料地短（`src/collectives.cc`，NCCL 2.27，NVTX 打点已删去）：

```c
// collectives.cc（NCCL 2.27，有删节）
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
```

第 3 篇列过的那张参数表，在这里原样装进一个 `ncclInfo` 结构体，然后交给 `ncclEnqueueCheck`（`src/enqueue.cc`）。顾名思义，这个函数做的是**入队**：把这次调用登记成 communicator 规划器（planner）里的一条任务。之后 NCCL 的流程分两段：

- **规划**：对每条任务，按消息大小选定算法、协议、channel 数和线程数（本篇第三、四、五节），组装成一个 kernel 计划（`ncclKernelPlan`）；
- **发射**：`ncclLaunchKernel` 把计划变成一次 `cuLaunchKernelEx`——就是第 3 篇 profiler 里那个 `ncclDevKernel_AllReduce_Sum_f32_RING_LL` 被排进队列的瞬间。

如果调用方用了 `ncclGroupStart/ncclGroupEnd`，规划会推迟到 group 结束时统一进行，一组任务可以合并发射——c10d 每次只登记一条任务，所以对它而言"入队、规划、发射"在一次 `ncclAllReduce` 调用内依次完成。group 的合并能力是 send/recv 路径的主角，下一篇讲。

把视角再拉远一层：从建组到执行，`ncclAllReduce` 牵涉的工作分属三个时机，本篇的内容就按这三个时机组织：

| 时机 | 发生时刻 | 干什么 | 结果存在哪 |
| --- | --- | --- | --- |
| 建组时 | 第一次使用这个组（第 3 篇的懒初始化） | 拓扑探测、环/树搜索、调优表计算、连接与缓冲建立 | communicator 内部（篇 11 说的通信子） |
| 调用时 | 每次 `ncclAllReduce` | 按消息大小查表，选算法、协议、channel 数、线程数 | kernel 计划与任务结构 |
| 执行时 | GPU 按队列排到 | kernel 沿环或沿树逐跳搬运并规约 | 你的输出张量 |

关键的分工是：**与"这次消息多大"无关的决策，全部在建组时固化；与消息大小有关的决策，留在调用时查表。**建组的那几百毫秒（第 3 篇实验一）买的是前者——拓扑要探测、环要搜索、连接要建立，每件都是全组协调的重活，不可能每次通信重做；调用时剩下的工作只有"量一下消息大小，查一次表"，开销在微秒级。第二节看建组时固化了什么，第三到第五节看调用时查的是什么表、执行时 kernel 怎么跑。

## 二、建组时：拓扑探测与环搜索

### 2.1 建组时固化的五件事

第 3 篇把建组概括为三步：发钥匙、入会、选路。现在打开 `initTransportsRank`（`src/init.cc`），把第三步"选路"展开成它实际做的五件事（有删节，函数名保留）：

```c
// init.cc（NCCL 2.27，有删节）：建组时，与拓扑相关的五步
ncclTopoGetSystem(comm, ...);          // ① 探测本机硬件，建成一棵 XML 树：
                                       //    CPU → PCIe 桥 → GPU（之间是 NVLink 还是 PCIe）、网卡
ncclTopoComputePaths(comm->topo, ...); // ② 计算任意两个节点之间的路径与速度
                                       //    （两卡间走 NVLink 还是绕 PCIe，网卡离哪张卡近）
ncclTopoCompute(topo, ringGraph);      // ③ 环搜索：在图上找出环，并决定能并行复制出几条
ncclTopoCompute(topo, treeGraph);      //    树搜索同理（collnet、nvls 各有一张图，略）
ncclTopoTuneModel(comm, ...);          // ④ 填调优表：bandwidths/latencies[原语][算法][协议]
// ⑤ 按搜索结果建立连接、分配缓冲——第 4 篇的 CUDA IPC 映射、
//    第 5 篇的内存注册与 QP 建立，都发生在这一步
```

五步的产物全部存在 communicator 里：一张拓扑图、每张算法图里的若干条环/树、两张调优表、以及到每个邻居的连接和缓冲。之后每次通信只是使用它们——这就是"建组一次、通信很多次"的含义。

第①步的 XML 树可以导出查看，环境变量是 `NCCL_TOPO_DUMP_FILE`。配合第 3 篇用过的调试日志，建组时的主要结论都能打印出来。

### 2.2 实验一：拓扑文件与环日志（exp25）

实验一不新增功能，只加观测：`exp25_topo.py` 与第 2 篇的 exp0 等价（初始化、一次 all_reduce、校验结果），目的是触发一次建组，真正的内容在环境变量里。

```python
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
```

```bash
NCCL_TOPO_DUMP_FILE=/tmp/topo.xml NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  torchrun --nproc_per_node=8 exp25_topo.py 2>&1 | tee /tmp/exp25.log
```

**先看拓扑文件。**`/tmp/topo.xml` 是第①步探测结果的文本形式，截取一台 8 卡 NVSwitch 机器的主要片段（字段较多，示意）：

```xml
<system version="1">
  <cpu numaid="0" affinity="..." arch="x86_64" vendor="GenuineIntel">
    <pci busid="0000:18:00.0" class="0x060400">
      <pci busid="0000:19:00.0" class="0x030200">
        <gpu name="NVIDIA A800-SXM4-80GB" dev="0" sm="80" rank="0" gdr="1">
          <nvlink target="0000:c7:00.0" count="6" tclass="0x068000"/>
          <nvlink target="0000:c8:00.0" count="6" tclass="0x068000"/>
        </gpu>
      </pci>
    </pci>
    ...
    <nic>
      <net name="mlx5_0" dev="0" speed="200000" port="1" .../>
    </nic>
  </cpu>
</system>
```

读法：每个 `<gpu>` 节点挂在它所属的 PCIe 桥之下，`nvlink` 子节点记录它到 NVSwitch 的链路数（这里 6+6=12 条），`<nic>` 记录网卡及其速率。第 1 篇岔路口"每条通信边独立选择通路"所需的信息——两卡之间有没有 NVLink、网卡在哪——就来自这棵树。异构机器（NVLink 只连相邻两卡、其余走 PCIe 的机器）上，这棵树不再对称，下面的搜索结果会明显不同，值得在有条件的机器上各导出一份对比。

**再看日志。**`NCCL_DEBUG_SUBSYS=INIT,GRAPH` 下，第③④步的结论以几行文本打印（8 卡示意，行数与数值以实测为准）：

```javascript
NCCL INFO Channel 00/04 :    0    5    2    7    3    6    1    4
NCCL INFO Channel 01/04 :    0    6    3    1    5    7    2    4
NCCL INFO Channel 02/04 :    0    1    6    5    3    2    7    4
NCCL INFO Channel 03/04 :    0    7    3    6    2    5    1    4
NCCL INFO Tree 0 : -1->0->1/-1/-1        ← 每个 rank 各打一行：父->自己->至多三个子
NCCL INFO Tree 1 : ...
```

（2 卡机器上环只有 `0 1` 两行，树退化为一条边——想看有内容的日志请用 4 卡以上。）第 3 篇读过的 `Ring 00 : 0 -> 1 -> 0` 与这里的 `Channel xx/xx` 是同一份信息在不同版本里的两种打印。

这三类行各回答一个问题：

- **`Channel xx/xx : ...`**：环搜索的结论。每行是一条 channel 上的环——channel 0 的环是 0→5→2→7→3→6→1→4→（回到 0）。注意各行的 rank 顺序不同：不同的 channel 走不同的环。
- **`Tree N : 父->自己->子/子/子`**：树搜索的结论。每个 rank 打印自己在第 N 棵树里的父节点和至多三个子节点（`-1` 表示没有）。每个 channel 实际建有**两棵**树，原因第三节讲。
- 还有一行在 GRAPH 子系统里：`Pattern 0, crossNic 0, nChannels 4, bw .../..., type NVL/...`——搜索结果的汇总：环图（pattern 0）、4 条 channel、机内链路类型 NVL（NVLink）。如果这行显示 `type SYS`，说明环被迫穿过了 PCIe/CPU——性能会差一个量级，这是排查通信慢时最先要看的行之一。

### 2.3 环是怎么搜出来的

`ncclTopoCompute`（`src/graph/search.cc`）做的事可以概括成一句：**在拓扑图上找一条经过所有 GPU 的环，使环上最慢那条链路的带宽最大；找到后，只要复制这条环还能增加总带宽，就再复制一条——每条复制品就是一条 channel。**

两点推论：

1. **channel 数是搜出来的，不是定的。**环图搜索的范围是 1 到 `MAXCHANNELS/2`（`MAXCHANNELS=64`，见 `device.h`），实际停在几条取决于复制是否还有收益。NVSwitch 机器上各条链路速率相同，复制几份总能摊出更多并行度；2 卡机器上只有一组链路，channel 数通常只有 1～2 条。实验四会看到 channel 数与性能的关系。
2. **各 channel 的环可以相同，也可以不同。**NVSwitch 全互联时链路没有区别，搜索出的各条 channel 往往是同一个环的复制（调优表里记作 `sameChannels`）；链路不均的机器上，不同的环让各 channel 错开使用链路，避免挤在同一条上。实验一日志里四行不同的 rank 顺序就是这么来的。

搜索完毕，每条环上的每一跳都要"选路"——两卡之间有 NVLink 就建 P2P 连接（第 4 篇的 CUDA IPC），只能经网络到达就建 NET 连接（第 5 篇的 QP 与注册内存）。第 3 篇日志里的 `Channel 00 : 0[0] -> 1[1] via P2P/IPC`，就是这一步给每条边记下的结论。到这里，第 1 篇那张岔路口图在代码里的位置就全部对上了：**岔路口不是一个 if 语句，而是建组时按路径类型选择 transport 的这段过程。**

---

## 三、算法：ring 与 tree

### 3.1 ring 的工业实现：把篇 2 的脚注读成代码

第 2 篇的原语表里有一条脚注：ring 实现里，all_reduce = reduce-scatter + all-gather。第 7 篇手写 mini-FSDP 时，这两个半程各自出现过一次。现在读 NCCL 的真身——ring 版 all_reduce 的 kernel 主循环，`runRing`（`src/device/all_reduce.h`，NCCL 2.27，有删节，注释为本篇所加）：

```c
// all_reduce.h（NCCL 2.27，有删节）
// 每张卡各跑一份；ringIx 是本卡在环上的序号；channelCount 是本 channel 分到的元素数
// prims 是"收发原语"对象，directXxx 一组方法表示机内直连（第 4 篇的显存直写）
for (elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  // ── 第一个半程：reduce-scatter，共 p-1 步 ──
  chunk = modRanks(ringIx - 1);
  prims.directSend(offset(chunk), nelem);            // 第 0 步：把指定块发给下一家
  for (j = 2; j < nranks; ++j) {                     // 中间 p-2 步：
    chunk = modRanks(ringIx - j);
    prims.directRecvReduceDirectSend(...);           //   收一块 → 与本地对应块规约 → 转发
  }
  chunk = ringIx;
  prims.directRecvReduceCopyDirectSend(..., postOp); // 第 p-1 步：最后一块规约到位，
                                                     //   写入本卡输出缓冲，同时转发给下一家

  // ── 第二个半程：all-gather，共 p-1 步 ──
  for (j = 1; j < nranks - 1; ++j) {                 // 中间 p-2 步：
    prims.directRecvCopyDirectSend(...);             //   收结果块 → 抄进输出缓冲 → 转发
  }
  prims.directRecv(...);                             // 第 2p-2 步：收齐最后一块
}
```

和第 7 篇的手写版对照着读，结构完全一致，差别在三个工程细节上：

1. **分块策略**。手写版把数据切成 p 块、每块一次收发；这里外面多了一层循环——本 channel 负责的数据先切成 p 个 chunk，chunk 太大时再分多轮（`loopCount`）流水推进，每一轮都是完整的两个半程。流水线化是为了让"等上一家"的时间被"发下一家"填满，链路不空转。
2. **每一步是三种动作的复合**。`directRecvReduceDirectSend` 一个调用完成"收、规约、转"三件事——规约（求和）就发生在转发的途中，不需要额外的缓冲区和中转 kernel。这是"通信即计算"在 kernel 里的写法。
3. **direct 前缀**。机内直连时，recv/send 最终落到对端显存的 load/store——第 4 篇亲手写过的那条 store，在这里是它的工业形态；跨机时同一位置换成经 FIFO 槽位的间接收发，由 proxy 线程接力（第 5 篇）。同一个 `runRing`，机内跨机两种走法，差别被封装在 prims 里——第 1 篇说的"判决对上层透明"，到 kernel 内部依然成立。

### 3.2 2(p-1)/p 的推导与带宽下界

第 3 篇第五节给过结论：ring all_reduce 每卡放到线上的字节是 `2(p-1)/p × 消息大小`，推导留到了本篇。现在补上。

设消息大小为 S（字节），共 p 张卡。数据被切成 p 块，每块 S/p。读上面的循环：

- **reduce-scatter 半程，p-1 步**：每一步每卡各发出一块、各收到一块。每卡共发出 (p-1)×S/p；
- **all-gather 半程，p-1 步**：同理，每卡再发出 (p-1)×S/p。

合计：**每卡发出（也收到）`2(p-1)/p × S`**。系数随 p 的变化：p=2 时是 1，p=8 时是 1.75，p→∞ 时趋于 2——**与卡数基本无关**，这就是 ring 能扩展到几千卡的原因（第 3 篇的原话）。

再说为什么这个数字同时是**下界**。任何 all_reduce 实现都绕不开两笔流量：本卡的那份数据必须被其余 p-1 张卡规约到，所以至少 (p-1)/p×S 要流出本卡；本卡要拿到完整的规约结果，至少 (p-1)/p×S 要流入本卡。ring 的每个半程恰好贴着这个下界走，没有一字节的多余传输。加上流水线把链路充满，大消息下 ring 的时间就逼近 `2(p-1)/p × S ÷ 链路带宽`——这就是"ring 能逼近 NVLink 带宽极限"的准确含义。

有一个指标专门用来度量这件事：nccl-tests 里的 **busbw**（总线带宽），定义为 `2(p-1)/p × S ÷ 实测时间`。它把"每卡实际上线的字节"摊回时间，可以直接和链路带宽比较——下面的实验就用它。

### 3.3 实验二：时间与卡数（exp26）

验证"时间与卡数基本无关"：固定消息 256 MiB，分别在 world=2、4、8 下计时。

```python
# exp26_ring_scale.py —— 实验二：固定消息大小，时间是否随卡数变化（GPU 专属）
# 分别用 torchrun --nproc_per_node=2 / 4 / 8 各跑一次
import time

import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验二：ring 的时间与卡数")
assert dev.type == "cuda", "本篇实验都需要 GPU/NCCL"

S = 256 * 2**20                       # 256 MiB
t = torch.ones(S // 4, device=dev)    # fp32

for _ in range(5):                    # 预热：建组与首次发射在计时之外
    dist.all_reduce(t)
torch.cuda.synchronize()

ts = []
for _ in range(20):
    torch.cuda.synchronize()          # 计时对齐（第 3 篇的常识）
    t0 = time.perf_counter()
    dist.all_reduce(t)
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
ts.sort()
t_med = ts[len(ts) // 2]
busbw = 2 * (world - 1) / world * S / t_med / 1e9   # 折算成 busbw，GB/s

if rank == 0:
    print(f"world={world}  消息 256 MiB  中位时间 {t_med * 1e3:.2f} ms  busbw ≈ {busbw:.0f} GB/s")
dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=2 exp26_ring_scale.py
torchrun --nproc_per_node=4 exp26_ring_scale.py
torchrun --nproc_per_node=8 exp26_ring_scale.py
```

预期输出（8 卡 NVSwitch 机器，示意，以实测为准）：

```javascript
world=2  消息 256 MiB  中位时间 2.10 ms  busbw ≈ 122 GB/s
world=4  消息 256 MiB  中位时间 2.15 ms  busbw ≈ 183 GB/s
world=8  消息 256 MiB  中位时间 2.25 ms  busbw ≈ 218 GB/s
```

三个数字各说明一件事。第一，时间近似不变——每卡上线字节 `2(p-1)/p × S` 从 1.0S 涨到 1.75S，但大消息下链路本就接近饱和，多出来的部分被并行吸收，这就是"与卡数几乎无关"的直接观察。第二，busbw 随 world 上升并逼近第 4 篇实测的 NVLink 单向带宽（约 260 GB/s）——world=2 时偏低，因为两卡之间只有一组链路，channel 少、流水线浅；8 卡时每卡有多条链路和多条 channel，利用率更高。第三，如果 world=8 的时间明显长于 world=2，先查实验一的日志：`Channel` 行有几行（channel 数是否够）、`type` 字段是不是 `NVL`（环是否被迫走了 PCIe）。

### 3.4 tree：用更少的步数换延迟

ring 的代价写在步数里：2(p-1) 步。p=1024 时，一次 all_reduce 要沿环接力 2046 跳——即使每跳不到一微秒，总延迟也已超过一毫秒。消息越小，带宽越无所谓，步数越刺眼。

tree 算法回答的就是步数问题：把 p 张卡组织成树，规约沿树上行、结果沿树下行，步数是 2·log 级别的，不再随 p 线性增长。NCCL 的树有两点具体设计值得知道：

1. **每棵树上，一个节点有一个父节点、至多三个子节点**（`NCCL_MAX_TREE_ARITY = 3`；实验一日志里 `Tree 0 : 父->自己->子/子/子` 一行的格式即由此而来）。树形由建组时的树搜索（`ncclTopoCompute`，pattern 为 BALANCED_TREE）根据拓扑决定，目标是让父子之间尽量落在高速链路上。
2. **每条 channel 建两棵树**，两棵树的父子关系错开，各承担一半数据——否则树中间的节点会成为瓶颈（一个节点的入向链路要汇聚多个子节点的流量）。两棵互补的树让每条链路上的负载重新拉平。

kernel 一侧（`runTreeUpDown`，同文件）分两段，结构与直觉一致：上行段，叶子直接 `directSend` 给父节点，中间节点 `directRecvReduceDirectSend`（收齐至多三个子节点的数据、与本地规约、发给父节点），根节点 `directRecvReduceCopy` 完成规约；下行段沿树广播回来。与 ring 相比，每个节点的"扇入"从 1 变成至多 3，这是模板参数 `FanAsymmetric<3, 1>` 的含义。

其余几种算法本篇只登记名字与适用条件，细节留到后面的篇目：**CollNet**（Direct/Chain 两种）把规约卸载到 IB 交换机的 SHARP 单元，需要交换机支持；**NVLS**（NVLink SHARP）在 NVSwitch 内完成规约，只有 NVSwitch 机器可选；**PAT** 面向"每节点一张卡"场景的 all-gather/reduce-scatter。它们是否可选，建组时探测后自动登记，实验一日志里若出现相关跳过提示属正常。

### 3.5 谁来选：建组时填表，调用时查表

ring 和 tree 怎么选？不是规则，是估价。建组时第④步（`ncclTopoTuneModel`）为每种（原语，算法，协议）组合填两个数：**延迟**（微秒）和**带宽**（GB/s），依据是一组内置的模型参数——摘录 ring 在 2.27 里的两行（`src/graph/tuning.cc`，单位微秒；跨机时换用网络那组）：

| 模型参数 | LL | LL128 | Simple |
| --- | --- | --- | --- |
| 基础延迟 `baseLat`（ring） | 6.6 | 14.0 | 8.4 |
| 每跳延迟（NVLink，ring） | 0.6 | 1.9 | 3.4 |

调用时（`enqueue.cc` 的 `updateCollCostTable`/`topoGetAlgoInfo`），对每个可用的（算法，协议）组合按实际消息大小估价，取最小者：

```c
// tuning.cc（NCCL 2.27，未删节）：估价模型 = 延迟 + 字节/带宽
*time = lat * latCount + nBytes / (1000 * bw);   // bw 单位 GB/s，结果单位 µs
```

这个式子本身平淡无奇，值得注意的是它把两件事分开了：**lat 和 bw 在建组时按硬件填好，调用时只剩一次乘加。**所谓"NCCL 自动选择"，落到代码里就是这张表加这一行式子。表本身可以打印（`NCCL_DEBUG_SUBSYS=TUNING`，建组时由 rank 0 输出每个组合的 lat/bw），环境变量 `NCCL_ALGO`、`NCCL_PROTO` 可以按原语粒度人工覆盖，语法支持排除写法，例如：

```bash
NCCL_PROTO="LL,Simple;allreduce:^LL"   # 全局允许 LL/Simple，但 allreduce 禁用 LL
NCCL_ALGO="ring,collnetdirect;allreduce:tree,collnetdirect;broadcast:ring"
```

（两行示例均出自 tuning.cc 的注释。）实验三会用到第一个变量。

---

## 四、协议：LL、LL128 与 Simple

### 4.1 三种协议的差别，是"旗标放在哪"

回顾一个前提：NCCL 的收发是异步的，发送方把数据放进约定位置后，接收方必须有一种办法知道"数据到了"。这个信号机制就是旗标（flag）——第 5 篇 FIFO 槽位里"数据到了插旗、旗到取货"的那个旗标。三种协议的本质区别只有一条：**旗标放在哪、多大粒度放一次。**

**LL（Low Latency）：旗标和数据写在一起，每 8 字节数据配 8 字节旗标。**数据结构是一个 16 字节的"线"（`src/include/device.h`，未删节）：

```c
// device.h（NCCL 2.27，未删节）
union ncclLLFifoLine {
  /* 旗标必须在数据之后：否则网络上可能出现"旗标先到、数据未到"。
     这里假定按 8 字节原子写入（IB/RDMA 满足），或连续整段到达（socket）。 */
  struct {
    uint32_t data1;
    uint32_t flag1;   // 4 字节数据 + 4 字节旗标
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};
```

旗标的值是当前步数（每收发一轮加一），接收方轮询旗标、对上步数即取数（`prims_ll.h` 里的 `readLL`，一条 `ld.volatile` 加一次比较）。旗标与数据同行，不需要单独的同步通道，延迟可以做到最低——这就是第 5 篇说的"数据和小旗写在一起，网卡一次 DMA、货和旗同时落地"。代价写在格式里：**16 字节里只有 8 字节是数据，带宽效率上限 50%。**

**LL128：同样的思路，粒度放大到 128 字节。**每条"线"128 字节，其中 120 字节数据、8 字节旗标，效率上限 120/128 ≈ 92%。粒度大、旗标少，带宽效率接近满格；但一次要等够 120 字节才发一轮，延迟高于 LL。另一个使用条件：缓冲地址需要 128 字节对齐。

**Simple：数据不带旗标，旗标单独走。**数据整块写进 FIFO 槽位（每连接 4 MiB 缓冲，8 个槽循环使用），收发的同步靠一个独立的 64 位步进计数：发送方写完一槽数据后更新对端的计数（`prims_simple.h` 里 `fence_acq_rel_sys()` 之后一次 `st_relaxed`），接收方轮询计数到位再读。数据通道上一个字节的额外开销都没有，带宽效率接近 100%；代价是"等一整槽才翻一次计数"，粒度最粗，延迟最高。

Simple 还有一个机内特例值得登记：当收发双方同机且显存可互访（第 4 篇的 CUDA IPC），NCCL 会把接收方**用户缓冲**的地址直接换给发送方（连接建立时的 `ptrExchange`），发送方 kernel 直接写进接收方的输出张量，连内部 FIFO 都省掉——第 4 篇亲手做的那次显存直写，在这里是它的最终形态。

三种协议摆在一起：

| 协议 | 旗标位置 | 带宽效率上限 | 特点 | 适用 |
| --- | --- | --- | --- | --- |
| LL | 与数据同行（每 8 B 数据 8 B 旗标） | 50% | 粒度最细，延迟最低 | 小消息 |
| LL128 | 与数据同行（每 120 B 数据 8 B 旗标） | ≈92% | 折中 | 中等消息 |
| Simple | 独立步进计数 | ≈100% | 粒度最粗，延迟最高 | 大消息 |

调优表里能直接看到前两行系数的出处（tuning.cc）：ring+LL 的带宽被打上 0.5 的折，ring+LL128 被打上 0.92（注释里写明 `120.0/128.0`）。模型参数不是凭空设的，是从协议的线格式推出来的。

### 4.2 实验三：协议扫描与切换点（exp27）

第 3 篇练习题 2 让你记录过协议后缀随消息大小的切换点，第 5 篇 `ib_write_bw` 的小消息拐点图是同一种现象的点对点版本。这一节把三种协议放在同一组尺寸下各测一遍，切换点的成因就直接可见了。

```python
# exp27_proto_sweep.py —— 实验三：同一组尺寸下三种协议的耗时（GPU 专属）
# 用法：
#   NCCL_PROTO=LL     torchrun --nproc_per_node=2 exp27_proto_sweep.py
#   NCCL_PROTO=LL128  torchrun --nproc_per_node=2 exp27_proto_sweep.py
#   NCCL_PROTO=Simple torchrun --nproc_per_node=2 exp27_proto_sweep.py
#   （不设 NCCL_PROTO 的一次运行是对照组：NCCL 自动选择）
import os
import time

import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验三：协议扫描")
assert dev.type == "cuda", "本篇实验都需要 GPU/NCCL"
proto = os.environ.get("NCCL_PROTO", "auto")

if rank == 0:
    print(f"协议 = {proto}（环境变量在建组时读取，每次运行只测一种）")

for exp in range(12, 29, 2):                 # 4 KiB, 16 KiB, ..., 256 MiB
    S = 2**exp
    t = torch.ones(S // 4, device=dev)       # fp32
    for _ in range(5):
        dist.all_reduce(t)                   # 预热
    torch.cuda.synchronize()
    ts = []
    iters = 50 if S <= 2**20 else 10
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(t)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    if rank == 0:
        print(f"  {S / 2**20:8.2f} MiB   中位 {ts[len(ts) // 2] * 1e6:9.1f} µs")
    del t
    torch.cuda.empty_cache()
dist.destroy_process_group()
```

预期输出（2 卡 NVLink 机器，示意，以实测为准；重点看三行的相对位置）：

```javascript
协议 = LL                协议 = LL128              协议 = Simple
    0.00 MiB   中位   7.9 µs      0.00 MiB   中位  17.5 µs      0.00 MiB   中位  15.2 µs
    1.00 MiB   中位  22.4 µs      1.00 MiB   中位  28.9 µs      1.00 MiB   中位  26.3 µs
   16.00 MiB   中位 280.0 µs     16.00 MiB   中位 240.0 µs     16.00 MiB   中位 170.0 µs
  256.00 MiB   中位5200.0 µs    256.00 MiB   中位3000.0 µs    256.00 MiB   中位2100.0 µs
```

把三次运行和不设变量的对照组画在一张图上，读出三个结论：

1. **小消息端 LL 最快。**4 KiB 时 LL 约 8 µs，LL128 和 Simple 都在 15 µs 以上——与 3.5 节模型表的每跳延迟（0.6 / 1.9 / 3.4 µs）次序一致。小消息的总时间几乎全是延迟，旗标粒度细的赢。
2. **大消息端 Simple 最快，LL 最慢。**256 MiB 时 LL 的时间约为 Simple 的两倍半——带宽上限 50% 对 100% 的直接体现；LL128 居中（92% 上限，另受每 channel 带宽上限约束）。
3. **对照组每一行都应贴近该行最优值。**auto 模式没有额外开销，它只是按消息大小查了那张表。把对照组的 kernel 名抓下来（profiler 或 `NCCL_DEBUG=INFO` 的 COLL 日志），就能看到切换点：小消息 `RING_LL`、中段 `RING_LL128`、大消息 `RING_Simple`。第 3 篇练习题 2 的切换点记录，在这里对上答案。

### 4.3 切换点为什么在那里

把 3.5 的估价式 `时间 = 延迟 + 字节数/带宽` 按协议各写一份，交点可以直接算出来。LL 与 Simple 的交点满足：

```text
lat_LL + S/bw_LL  =  lat_Simple + S/bw_Simple
=>  S* = (lat_Simple - lat_LL) × (bw_LL × bw_Simple) / (bw_Simple - bw_LL)
```

代入模型表的典型值（NVLink、ring、小 p）：延迟差约 7～8 µs，bw_LL 约为 bw_Simple 的一半，算出交点在几十到一百多 KB 量级——与实测的切换点一致。LL128 与 Simple 的交点同理，落在 MB 量级。**切换点不是调出来的常数，是"延迟差"与"带宽差"两条线的交点**；换一台机器（链路带宽变了、跳数变了），表里的 lat/bw 跟着变，交点自己移动。这也是 NCCL 不把切换点写死、而建组时按硬件重算一遍的原因。

---

## 五、channel：并行搬运的通道数

### 5.1 一条 channel = kernel 的一个线程块

channel 这个词在第 3 篇的日志里就出现过（`Channel 00 : ...`），一直没有正式定义。现在可以给全了：**一条 channel 是环（或树）的一份独立拷贝，连同它自己的连接、缓冲和旗标；数据被切成 channel 份，各条 channel 并行地沿各自的环搬运。**

它在 kernel 发射处的体现只有一行（`enqueue.cc` 的 `ncclLaunchKernel`，有删节）：

```c
// enqueue.cc（NCCL 2.27，有删节）
int nChannels = countOneBits(plan->channelMask);
dim3 grid  = {(unsigned)nChannels, 1, 1};   // 一个线程块负责一条 channel
dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
```

一次集合通信发射的 kernel，其 grid 恰好是 channel 数个线程块——第 11 篇说"通信 kernel 只占很少几个线程块"，数量就是这里的 channel 数。块内线程数 `threadPerBlock` 按协议定：Simple 默认 512（ring 额外加一个 warp 负责同步），LL 为 512，LL128 为 640，上限 640（`NCCL_MAX_NTHREADS`）。每条 channel 占一个 SM 的一份线程块档期，SM 占用极少——这是第 7 篇通信与计算能够重叠的硬件基础。

任务结构里记着本次通信覆盖的 channel 区间和每条 channel 的字节数（`device.h`，有删节）：

```c
// device.h（NCCL 2.27，有删节）
struct alignas(16) ncclDevWorkColl {
  uint32_t channelLo:8, channelHi:8;   // 覆盖的 channel 区间 [lo, hi]
  uint32_t nWarps:8;                   // 每条 channel 用多少个 warp
  ...
  struct {
    size_t countLo, countMid, countHi; // 首、中、尾 channel 各分到的字节数
    ...
  } cbd;                               // 按连续字节切分：channel c 拿第 c 段
};
```

切分方式是**连续字节切分**：消息按字节排成一条线，channel 0 拿第一段、channel 1 拿第二段……（首尾的 channel 允许与中间不等长，以凑齐对齐）。每条 channel 内部再按环切成 p 个 chunk，走 3.1 节那两个半程。两层切分叠起来：先按 channel 分、再按 rank 分。

这里需要消除一个同名歧义：第 11 篇 §2.1 的"硬件通道（channel）"是 GPU 驱动下发命令的队列通道；本篇的 channel 是 NCCL 的并行搬运通道。两者正交——NCCL kernel 作为一个任务经某条硬件通道下发，kernel 内部再按 channel 分成若干线程块。

### 5.2 channel 数的两层决定

channel 数在两个时机各决定一次：

- **建组时给上限。**环搜索（2.3 节）按"复制环是否还增加总带宽"定出 channel 上限，写进 communicator。环境变量 `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` 可以钳制这个范围（旧版的单一变量 `NCCL_NCHANNELS` 等价于两个同值）。
- **调用时按消息大小收缩。**消息太小时，channel 多了反而每条分到的数据太碎、固定开销摊不掉，于是调用时按一个经验规则递减（`enqueue.cc`，有删节）：

```c
// enqueue.cc（NCCL 2.27，有删节）：消息越小，channel 越少
while (nBytes < nc * nt * threadThreshold) {
  if (nc >= 2) nc--; else break;
}
// 线程数同理：消息小则 warp 数按 128 线程一档向下减半
```

`threadThreshold` 是每个（算法，协议）组合的"每线程最少字节数"阈值，建组时填好（LL 还要乘上 rank 数）。两条收缩规则合起来的含义：**小消息集中火力（少 channel、少线程），大消息全面铺开。**所以小消息的 kernel 可能只带 1～2 个线程块，SM 占用比大消息时更小。

### 5.3 实验四：channel 数扫描（exp28）

把 channel 数锁成固定值，测大消息（1 GiB）的时间随 channel 数的变化：

```python
# exp28_channels.py —— 实验四：固定 channel 数，测大消息耗时（GPU 专属）
# 用法（env 在建组时读取，每组 channel 数需要单独启动一次）：
#   NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 torchrun --nproc_per_node=8 exp28_channels.py
#   NCCL_MIN_NCHANNELS=2 NCCL_MAX_NCHANNELS=2 torchrun --nproc_per_node=8 exp28_channels.py
#   ... 依次换 4/8/16/32
import os
import time

import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验四：channel 数扫描")
assert dev.type == "cuda", "本篇实验都需要 GPU/NCCL"
nc = os.environ.get("NCCL_MAX_NCHANNELS", "auto")

S = 2**30                                # 1 GiB
t = torch.ones(S // 4, device=dev)
for _ in range(3):
    dist.all_reduce(t)
torch.cuda.synchronize()

ts = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(t)
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
ts.sort()
t_med = ts[len(ts) // 2]
busbw = 2 * (world - 1) / world * S / t_med / 1e9

if rank == 0:
    print(f"channels={nc:>4}  1 GiB  中位 {t_med * 1e3:7.2f} ms  busbw ≈ {busbw:.0f} GB/s")
dist.destroy_process_group()
```

预期输出（8 卡 NVSwitch 机器，示意，以实测为准）：

```javascript
channels=   1  1 GiB  中位   32.50 ms  busbw ≈  58 GB/s
channels=   2  1 GiB  中位   16.90 ms  busbw ≈ 111 GB/s
channels=   4  1 GiB  中位    9.10 ms  busbw ≈ 206 GB/s
channels=   8  1 GiB  中位    8.60 ms  busbw ≈ 218 GB/s
channels=  16  1 GiB  中位    8.55 ms  busbw ≈ 219 GB/s
channels=  32  1 GiB  中位    8.60 ms  busbw ≈ 218 GB/s
```

读法：channel 数从 1 翻倍到 4，时间近似减半——多条 channel 把环复制成多份并行搬运，busbw 跟着线性涨；到 8 条以后进入平台，链路带宽被打满，再加 channel 没有收益。busbw 的平台值应与实验二 world=8 的数字一致（同一台机器、同一个上限），与第 4 篇实测的链路带宽（约 260 GB/s）之间的差距，是协议开销与流水线填充的真实成本。

把消息改小（比如 1 MiB）重跑，会看到相反的关系：channel 越多越慢——每条 channel 分到的数据太少，固定开销（发射、旗标轮转、线程调度）占了主导。这正是 5.2 节"按消息大小收缩"存在的原因，也是"NCCL_NCHANNELS 调优"这个常见建议的适用边界：**只有大消息、且自动值偏保守时，手动加 channel 才有收益。**

---

## 六、收尾：三个问题的答案

回到开头的三个问题。

**第一问：算法、协议、channel 各在哪一步决定。**算法与协议在**调用时**决定——`ncclAllReduce` 把参数装进 `ncclInfo` 入队后，NCCL 对每个（算法，协议）组合用"延迟 + 字节数/带宽"估价，取最小者；估价的依据（每种组合的延迟与带宽）连同环、树、连接、缓冲，全部在**建组时**探测、搜索并固化进 communicator。channel 数两次决定：建组时环搜索给上限，调用时按消息大小向下收缩。三步落到代码里就是三个文件：`init.cc`（建组）、`tuning.cc`（估价表）、`enqueue.cc`（选择与发射）。

**第二问：ring 为什么逼近带宽上限。**ring 把 all_reduce 拆成 reduce-scatter 与 all-gather 两个 p-1 步的半程，每卡上线字节恰为 2(p-1)/p × S——这个数字同时是 all_reduce 的流量下界，ring 没有一字节多余传输；分块流水线把链路充满后，大消息时间贴着"下界流量 ÷ 链路带宽"走。实验二里时间与卡数基本无关、busbw 逼近链路带宽，是这个推导的直接观察。

**第三问：三种协议怎么切换。**差别只有一处——旗标的位置与粒度：LL 每 8 字节数据配 8 字节旗标（带宽上限 50%，延迟最低），LL128 每 120 字节数据配 8 字节旗标（上限 92%），Simple 用独立的步进计数（上限约 100%，延迟最高）。NCCL 不写死切换点，而是建组时按硬件填好每种组合的延迟与带宽，调用时估价取最小；切换点因此随机器自动移动，实验三强制三种协议各跑一遍，能把这条选择曲线完整描出来。

本篇读了 NCCL 的一半：集合通信从入队到 kernel 的完整链路。剩下两半在下一篇：send/recv（篇 8 流水线并行的底层，走 P2P work 而非环/树，ncclGroupStart/End 的合批在这里生效），以及 proxy 线程的推进循环（第 5 篇只画了它与 kernel、网卡的分工，下一篇读它的代码）。

---

## 练习题

（按难度排序，欢迎把结果贴在评论区）：

1.（计算）由 LL 的线格式（16 字节含 8 字节数据）推算它的带宽上限；LL128 同理。为什么调优表里 ring+LL 的带宽恰好打 0.5 的折、ring+LL128 打 0.92？
2.（计算）p=64 时，ring 每卡上线字节的系数是多少？与 p=8 相差多少个百分点？这个差值说明了 ring 对卡数的什么性质？
3.（概念）建组时与调用时各决定什么？把拓扑搜索放在建组时的理由是什么？（提示：想想第 3 篇实验一那几百毫秒花在哪，以及它为什么不能花在每次通信上。）
4.（读表）实验二里，如果 world=8 的中位时间明显长于 world=2（比如 2 倍以上），最可能的原因出在哪一层？按什么顺序查实验一的日志？（提示：先看 Channel 行数，再看 type 字段。）
5.（推断）5.2 节的收缩规则对小消息减少 channel，而不是增加。从"每条 channel 的固定开销"和"数据粒度"两个角度说明原因。
6.（动手）在实验三的对照组（不设 `NCCL_PROTO`）里用第 3 篇的 profiler 抓 kernel 名，标出 LL→LL128→Simple 的两个切换点，与 4.3 节算出的交点对照；再与你在第 3 篇练习题 2 里记录的数字对照。
7.（动手）实验三加跑 `NCCL_ALGO=Tree`，与 ring 的结果放在同一张图上：在你的机器上，tree 在哪些尺寸段占优？用 3.5 节模型表里 tree 与 ring 的每跳延迟差解释。
8.（推断）`MAXCHANNELS=64`，而环搜索的上限是其一半（32）。channel 为什么不能无限多？从 SM 占用（第 11 篇）、每连接缓冲开销、链路带宽收益三个角度各给出一条理由。

---

## 附录 A：报错锦囊

| 症状 | 原因与处置 |
| :--- | :--- |
| `NCCL_TOPO_DUMP_FILE` 没有生成文件 | 文件在建组（第一次使用这个组）时生成；默认只有 rank 0 写出（`NCCL_TOPO_DUMP_FILE_RANK` 可改）；确认环境变量传进了子进程 |
| 日志里没有 Channel/Tree 行 | 需要 `NCCL_DEBUG=INFO` 且 `NCCL_DEBUG_SUBSYS` 包含 INIT,GRAPH；日志走 stderr，管道前加 `2>&1`（第 3 篇附录同款） |
| 设了 `NCCL_PROTO=LL` 后大消息明显变慢 | 预期现象：LL 带宽上限减半，正是实验三要观察的内容，不是故障 |
| `NCCL_NCHANNELS` 似乎不生效 | 新版拆成 `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS` 两个变量，只设一个只是钳制范围，两个设同值才锁死；变量在建组时读取，启动后修改无效 |
| kernel 名里的字段顺序与本篇不同 | 版本差异（第 3 篇附录同款）；认出"算法/协议/规约/类型"四类字段即可 |
| 实验三三种协议耗时几乎相同 | 尺寸范围太小，或 2 卡链路过于富余；把尺寸上限加大、world 加到 8 再看 |
| 日志出现 NVLS/CollNet 被跳过的提示 | 二者分别需要 NVSwitch（NVLS）与 IB SHARP（CollNet）支持；不满足时 NCCL 自动不选，属正常 |
| `type SYS` 出现在环图汇总行 | 环被迫穿过 PCIe/CPU 互连，通信带宽降一个量级；检查卡间 NVLink 连接与 `CUDA_VISIBLE_DEVICES` 的设置 |

## 附录 B：本篇文件清单

```text
数据盘/tp-lab/
├── mini_tp.py              # 第 2 篇起沿用
├── exp0_hello.py           # 第 2 篇：实验一的等价前身
├── exp25_topo.py           # 本篇：触发建组，配合 NCCL_TOPO_DUMP_FILE 与 INIT,GRAPH 日志
├── exp26_ring_scale.py     # 本篇：固定 256 MiB，world=2/4/8 各跑一次，验证 2(p-1)/p
├── exp27_proto_sweep.py    # 本篇：NCCL_PROTO 强制三种协议，尺寸从 4 KiB 扫到 256 MiB
└── exp28_channels.py       # 本篇：NCCL_MIN/MAX_NCHANNELS 锁定 channel 数，测 1 GiB 大消息
```

四个脚本均为 GPU/NCCL 专属，运行方式照旧 `torchrun --nproc_per_node=N 脚本名.py`；实验二需要多档 N 各跑一次，实验三、四的环境变量必须在启动前设置。素材：`collectives.cc`、`enqueue.cc`、`graph/tuning.cc`、`graph/search.cc`、`device/all_reduce.h`、`include/device.h`（NCCL 2.27，随 torch 2.8 自带；源码仓库 NVIDIA/nccl）。

## 下篇预告

《NCCL 内部（下）：send/recv 与 proxy 推进引擎》——集合通信这一半读完，NCCL 还剩另外两半。一是第 8 篇流水线并行用的 `dist.send`/`dist.recv`：它在 NCCL 里不长成环或树的 kernel，走的是另一条 P2P 任务路径，`ncclGroupStart/End` 把一批收发合并进一次发射的机制也在这里。二是第 5 篇只画了分工的 proxy 线程：它的推进循环怎么轮询 FIFO、怎么下发 verbs、信用怎么流转，以及它出错时第 11 篇的看门狗看到的是什么。读完这两半，第 1 篇那张七层图的 L5 就全部走完。

---

> 文章合集仓库：`github.com/1989chenguo/LLM-Networking-Stack`，欢迎 star、issue 交流。
