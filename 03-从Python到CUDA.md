# 从Python到CUDA：一行 all_reduce 是怎么变成 GPU kernel 的

> 连载第 3 篇 · 从Python到CUDA
>
>
> 上一篇把黑盒的边界划在 `dist.all_reduce`：线以上——进程、rank、组、原语——已经全部讲完。这一篇往下走第一层：看这行调用是怎么"下去"的。
>
> 主角还是上一篇那个 6 行的 exp0，一行不用改，只加三件观测工具：计时器、profiler、NCCL 的调试日志。走完这一篇，从 `dist.all_reduce` 到 GPU kernel（在 GPU 上执行的那段程序，第三节会正式认识它）之间的每一站，你都有亲眼见过的证据。
>
> 本篇会停在一个新问题前：NCCL 的日志说它选了"显存直写"——进程都是隔离的，凭什么写别人的显存？那是下一篇的事。

**本篇的三个问题**

1. 第一次 `all_reduce` 为什么比后面的慢几个数量级？
2. Python 的一行调用，是怎么变成 GPU 上的一个 kernel 的？
3. NCCL 日志里的 `via P2P/IPC` 是什么意思？（本篇只负责读到这句话，"什么意思"是下一篇的事）

**动手指南**：本篇有两个实验脚本，运行方式照例：

```bash
torchrun --nproc_per_node=2 脚本名.py
```

---

## 一、实验一：第一次为什么慢

给 exp0 的 all_reduce 加个计时器，连跑 6 次。新建 `exp3_timing.py`：

```python
# exp3_timing.py —— 实验一：第一次 all_reduce 为什么慢
import time
import torch
import torch.distributed as dist
from mini_tp import lab_init

dev, rank, world = lab_init("实验一：第一次为什么慢")
t = torch.ones(1024, 1024, device=dev)   # 4 MB 的 fp32

for i in range(6):
    if dev.type == "cuda":
        torch.cuda.synchronize()         # 计时对齐：等 GPU 把排队的活干完
    t0 = time.perf_counter()
    dist.all_reduce(t)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    if rank == 0:
        print(f"第 {i} 次 all_reduce：{(time.perf_counter() - t0) * 1e3:.2f} ms", flush=True)
dist.destroy_process_group()
```

```bash
torchrun --nproc_per_node=2 exp3_timing.py
```

预期输出（数字以实测为准，量级是关键）：

```javascript
第 0 次 all_reduce：432.71 ms
第 1 次 all_reduce：0.09 ms
第 2 次 all_reduce：0.08 ms
第 3 次 all_reduce：0.08 ms
...
```

第一次和后面差了**三四个数量级**。先看清楚哪部分才是"通信本身"：4 MB 数据在 NVLink 上搬运，理论时间是几十微秒量级——所以第 1 次往后的 0.1 ms 才是 all_reduce 的真身；第 0 次多出来的几百毫秒，与消息大小无关（练习题 1 会验证这一点），是一次性成本。

这个成本的名字叫**懒初始化**：`init_process_group` 只建立了上一篇说的"组员名单"，真正的通信通道是**第一次用到这个组时才建**的。为什么要懒？真实框架启动时会建几十个子组（Megatron 的 TP/DP/PP 组矩阵），但不是每个组在每个阶段都用；懒初始化保证只有真用到的组才付建组成本（练习题 4）。至于这条"通道"是谁建的、建的时候干了什么——正是下一节的内容。

注意一个计时细节，是以后会反复用到的常识：

- `all_reduce` 是**异步派发**的：CPU 把活派给 GPU 就返回。不做 `synchronize`，你测到的只是 CPU 的返回时间——附录 A 有这条报错对照；

## 二、那几百毫秒花在哪：两个新朋友，建组三步

### 2.1 先认识 c10d 和 NCCL

那几百毫秒里要登场两个新角色，先正式介绍。

**c10d**：PyTorch 自带的 C++ 分布式库。你在 Python 里写的 `dist.all_reduce`，真正干活的是它——上一篇讲的进程组、原语，在代码层面都是 c10d 的对象和函数。源码在 pytorch/pytorch 仓库的 `torch/csrc/distributed/c10d/` 目录，第三节会读其中一段。

**NCCL**（NVIDIA Collective Communications Library）：NVIDIA 开源的集合通信库，一个独立的 C 库，torch 官方包自带，不用单独装。亲眼确认它就在你机器上：

```bash
python -c "import torch; print(torch.cuda.nccl.version())"   # 打印类似 (2, 27, 3)
```

本篇会出现的 `ncclGetUniqueId`、`ncclCommInitRank`、`ncclAllReduce` 都是 NCCL 的公开 API——在官方文档（docs.nvidia.com/deeplearning/nccl）和头文件 `nccl.h`（源码仓库 NVIDIA/nccl）里都查得到，不是什么内部魔法。

### 2.2 建组三步

第一次调用里，这两家合力做了三件事（每步标出主导方）。三步里有三个名词会反复出现，边用边解释：**communicator**、**bootstrap socket**、**通信环**。

1. **发钥匙（c10d 主导）**。先解释 communicator 是什么：上一篇的"组"是 c10d 层的名册，而 **communicator（通信器）是这个组在 NCCL 层的实体**——可以把它想成全组共用的一部对讲机，里面装着通信所需的全部家当：成员名单、和邻居的连接、缓冲区，以及第三步要排的队形。之后每次 all_reduce，都是"拿着这部对讲机说事儿"。NCCL 规定每部对讲机要配一把"钥匙"：一个 128 字节的 `ncclUniqueId`，全组拿同一把才能认出彼此。生成钥匙是 NCCL 的 API（`ncclGetUniqueId()`），但"谁生成、怎么发到全组"是 c10d 安排的——让组内 rank 0 生成，借上一篇那块 TCPStore 签到板广播给全组。注意，**慢通道交换钥匙、快通道才搬数据**——这个套路后面还会遇到。
2. **入会（NCCL 主导）**。每个进程调 `ncclCommInitRank(nranks, uniqueId, myrank)`，凭钥匙入会。入会时 NCCL 还会自己拉起一套 **bootstrap socket**——名字唬人，其实就是几条普通的 TCP 连接，专职交换"开门用的信息"（比如下一篇要讲的显存句柄），不搬真正的数据。这是"慢通道换钥匙"在本篇的第二次出现。
3. **选路（NCCL 主导）**。拓扑探测：NCCL 摸清这台机器上哪两块卡之间有 NVLink、PCIe 怎么连、网卡挂在哪，然后把全组排成一个环——**通信环（ring）**是 NCCL 的默认搬运队形：数据沿环流动，每人只和左右两个邻居打交道。为什么排成环、而不是"人人都发给 rank 0"？第 2 篇练习题 4 你已经亲手体会过了：那样 rank 0 会被压垮，而环上每个人的负载和人数无关（另一种队形是树，深挖篇目再讲）。排好环之后，NCCL 给环上的**每一条边**挑一条路。判决结果是什么样，第六节读日志就看到了。

三步做完，communicator 就绪：钥匙、邻居连接、环形队形全部就位。之后的每次 all_reduce 就只是"沿现成的环搬数据"——这才是那 0.1 ms 的全部内容。

## 三、c10d 的一段源码：所有原语共用的 collective()

上一节说过，`dist.all_reduce` 真正干活的是 c10d。读它一段真代码：`ProcessGroupNCCL::allreduce`（`torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`）并不自己干活，而是转头调用一个公共函数 `collective()`，只把一句 NCCL 调用塞给它：

```cpp
// ProcessGroupNCCL.cpp（主线版本，有删节）
c10::intrusive_ptr<Work> ProcessGroupNCCL::allreduce(...) {
    return collective(                       // 所有集合通信共用的流水线
        inputs, outputs, ...,
        [&](at::Tensor& in, at::Tensor& out,
            ncclComm_t comm, at::cuda::CUDAStream& stream) {
            return ncclAllReduce(in.data_ptr(), out.data_ptr(), in.numel(),
                                 ncclDataType, ncclSum, comm, stream);
        },
        ...);
}
```

为什么需要一个"公共"函数？因为 allreduce、allgather、reduce_scatter、broadcast……这些原语走进 c10d 后，要做的杂活一模一样——检查张量、建组、排队、发工单——**只有最后那句"调 NCCL 的哪个函数"不同**。所以公共流程写进 `collective()`，每个原语只填自己那句 NCCL 调用：上面代码里 `[&](...) { return ncclAllReduce(...); }` 这一段，就是 allreduce 填的部分。（这个函数在 C++ 里恰好也写成函数模板 template，算个双关；理解成"公共流水线"就够了。）

这条流水线长什么样？把 `collective()` 展开成示意骨架（删节自 ProcessGroupNCCL.cpp，真实函数名以你 checkout 的版本为准）：

```cpp
// collective() 的示意骨架：杂活都在这里，四件事的位置标在注释里
collective(..., Fn fn, ...)              // fn：各原语填进来的那句 NCCL 调用
{
    check_gpu_tensors(...);              // ① 检查张量：连续、同设备、dtype 合法
    auto comm = getNCCLComm(...);        // ② 懒初始化：没有 communicator 就现场建组（第一、二节）

    // ③ stream 对齐（实际是 CUDA event 的 record/wait 配对）：
    ncclStream.waitEvent(计算流的进度);   //    NCCL 队列先等计算队列干完
    fn(input, output, comm, ncclStream); //    开工：执行填进来的那句 ncclAllReduce
    计算流.waitEvent(通信的进度);         //    通信完记一笔，计算队列再等它

    return WorkNCCL(...);                // ④ 打包成工单返回，watchdog 线程后台盯梢
}
```

四件事不是我编的，就是这条流水线里实打实的四步。各自是什么意思——其中有三个新名词（stream、kernel、Work 句柄），先解释再对号：

- **stream（流）** 是 GPU 上的任务队列。你写的每个 PyTorch 算子（矩阵乘、加法）都被 CPU 排进队列，GPU 按顺序干活，CPU 排完就返回、不等待——这就是第一节说的"异步派发"。NCCL 的通信也是一个 GPU 任务，排进它自己专用的队列。第③步干的就是两条队列的对齐：先等你算完，再通信；通信完了，你再接着算。
- **kernel（核函数）** 就是排进队列的那些任务的本体：一段在 GPU 上执行的函数。和 Python 函数不同，它不是一个线程按部就班地跑，而是 GPU 上成千上万个线程一起跑——一次矩阵乘被拆成几万份同时算。你的每个算子是一个 kernel，NCCL 的一次通信也是一个 kernel。CPU 的角色只是"派活"：把 kernel 发射（launch）进队列就返回，GPU 按队列顺序执行。
- **Work 句柄** 是一张"工单号"。通信是异步的，调用发起时活还没干完，c10d 把这次调用打包成一个对象返回——拿着它，可以查状态、可以等它完成（后面 Megatron 篇目会看到 `async_op=True` 返回的就是它）。

四件事就好懂了：

- **① 检查张量**：来料检验，不合格当场报错；
- **② 懒初始化**：就是第一、二节讲的建组；
- **③ stream 对齐**：上面刚讲——`all_reduce` "异步"的根源就在这两条队列的分工；
- **④ 发工单**：打包成 Work 句柄返回，后台的 **watchdog（看门狗）**线程盯着每张工单，超时没完成就报错——上一篇说"集合通信少一个人就挂起、超时后报错"，执行者就是它。

最后看 allreduce 填进流水线的那句 `ncclAllReduce`。注意它和第二节讲的是两类 API：第二节的 `ncclGetUniqueId`、`ncclCommInitRank` 是**建组**的 API，而它是**开工**的 API——建组是"装好对讲机"，`ncclAllReduce` 是"按下通话键"。再看它的参数表，每个你都已经认识了：

| 参数                            | 含义             | 你在哪见过它                    |
| ------------------------------- | ---------------- | ------------------------------- |
| `in.data_ptr(), out.data_ptr()` | 源地址、目标地址 | 你的 tensor `t` 的显存地址      |
| `in.numel()`                    | 元素个数         | `t` 里有多少个数                |
| `ncclDataType`                  | 数据类型         | `t` 的 dtype                    |
| `ncclSum`                       | 规约操作         | 第 2 篇说的"数学要件"：`op=SUM` |
| `comm`                          | communicator     | 2.2 节建好的那部"对讲机"        |
| `stream`                        | 任务队列         | 本节刚讲的 NCCL 专用队列        |

NCCL 拿着这些决定搬运方案——沿环还是沿树、分几路并行搬（channel）、用什么协议——然后生成一个 GPU kernel 去执行。

一句话记住分工：**c10d 翻译兼管家，NCCL 出搬运方案**。

## 四、实验二：在 profiler 里亲眼看见 kernel

"变成 GPU kernel"可以直接看。用的工具是 PyTorch 自带的 **profiler（性能记录仪）**：打开它，`with` 块里发生的每一件事——CPU 上调了什么、GPU 上跑了什么、各花了多少时间——都会被记在案。新建 `exp4_profiler.py`：

```python
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
```

```bash
torchrun --nproc_per_node=2 exp4_profiler.py
```

**先教读表**。profiler 打印的不是函数调用栈，而是一张"活动记账表"：每行一件被记下的活动，第一列是活动的**标签**，后两列是它自身在 CPU 上、在 GPU 上各花了多少时间。预期输出（截取关键行，示意；`...` 处你实测时会看到真实数字，字段顺序随 torch/NCCL 版本略有差异）：

```javascript
---------------------------  ------------  ------------
Name                         Self CPU      Self CUDA
---------------------------  ------------  ------------
nccl:all_reduce                ... μs         0.000 μs
ncclDevKernel_AllReduce_RING_LL_Sum_float   0.000 μs    ... μs   ← 真身
cudaLaunchKernel               ... μs         0.000 μs
cudaStreamSynchronize          ... μs         0.000 μs
```

逐行看：

- `nccl:all_reduce`：这次集合通信的 CPU 侧记录。注意它是 c10d 贴上去的**标签**，不是源码里的函数名——你在第三节看到的真名是 `ProcessGroupNCCL::allreduce` → `collective()`；profiler 不显示 C++ 源码符号，只显示这条贴好的艺名。
- `ncclDevKernel_AllReduce_RING_LL_Sum_float`：真正在 GPU 上跑的 kernel——第三节说的"排进队列的任务本体"，在这里露出了真名。**名字本身是一份履历**——`RING` 是算法、`LL` 是协议、`Sum_float` 是规约操作和数据类型。
- `cudaLaunchKernel`：CPU 把 kernel 发射进队列的那个动作——第三节说的"派活"，在 profiler 里留下了指纹。

**为什么一半的时间是 0？** 不是没干活，恰恰是分工的证据：CPU 侧的活动（发起调用、发射 kernel）不在 GPU 上花时间，GPU 列必然是 0；GPU 侧的 kernel 是 CPU 发射完就走人的，CPU 列必然是 0。一张表里 0 的分布，正好画出"CPU 派活、GPU 干活"的分界线——第三节讲的异步派发，在这里成了看得见的数字。

一次 Python 调用，最终长成一个带齐全部参数的 GPU kernel。协议会随消息大小自动切换（练习题 2 让你找出切换点）。

`--device cpu` 下这个实验看不到 `ncclDevKernel`：gloo 是 CPU 线程搬数据，本来就不经过 GPU kernel。这个差别本身就是"语义与传输分离"的又一证据——上一篇的原语义不变，变的只是底下的搬运方式。

**你现在的状态应该是**：亲眼见过第一次调用的几百毫秒、profiler 里的 `ncclDevKernel`、kernel 名里的算法/协议/规约三个字段。

## 五、ring 的一句话版本

kernel 名里的 `RING` 值得一句话：ring allreduce 把"求和并广播"拆成两个半程——先沿环做 reduce-scatter（每片数据在环上转一圈、求和到位），再沿环做 all_gather（把各片拼全）。每个 rank 放到线上的字节是 `2(p-1)/p × 消息大小`——**与卡数几乎无关**，这是它能扩展到几千卡的根本原因。第 2 篇说过"reduce_scatter 是 all_reduce 的下半身"，现在你知道这句话的出处了。公式的推导和带宽实测，留给 NCCL 深挖篇目。

## 六、把建组日志读一遍：停在 via P2P/IPC

最后一件观测工具：NCCL 自己的调试日志。把建组过程打出来：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P \
  torchrun --nproc_per_node=2 exp0_hello.py 2>&1 | grep -E "Ring|via|comm" | head -10
```

预期输出（字段随 NCCL 版本略有差异，示意）：

```javascript
NCCL INFO comm 0x55... rank 0 nranks 2 ...       ← 入会完成：拿到组和编号
NCCL INFO Ring 00 : 0 -> 1 -> 0                  ← 环：数据沿它流动
NCCL INFO Channel 00 : 0[0] -> 1[1] via P2P/IPC  ← 选路结果
```

三行各自对应第二节建组三步的成果：入会是 step 2；Ring 和 Channel 是 step 3 选路的判决书——`0 -> 1 -> 0` 说两卡组成一个环，数据沿环流动。

重点是最后一行的 `via P2P/IPC`：NCCL 探测到两张卡之间有 NVLink，给这条边选的路是 **P2P/IPC——"显存直写"**。("新版 NCCL（配新驱动）这里会显示 P2P/CUMEM，与 P2P/IPC 同为显存直写，只是换钥匙的机制从 cudaIpc 换成了 cuMem；下一篇亲手做直写时，两种句柄都能完成"。)

本篇到此为止。停下来想一想这句话有多奇怪：你的两个进程是 torchrun 拉起来的两个独立进程，内存互相隔离——**进程 0 凭什么能"直写"进程 1 的显卡？**这个"P2P/IPC"到底是什么动作？下一篇不开 NCCL，亲手把这个动作做一遍。

---

## 七、收尾

回到开头的三个问题：

1. **第一次为什么慢**：慢的几百毫秒是懒初始化的建组成本——发钥匙（uniqueId 经 TCPStore 广播）、入会（ncclCommInitRank）、选路（拓扑探测），与消息大小无关；之后的每次调用才是在现成通道上搬数据。
2. **怎么变成 kernel 的**：c10d 的公共流水线 `collective()` 做翻译和管家（检查、懒初始化、stream 对齐、Work 句柄），NCCL 出搬运方案（ring、channel、协议），最终长成一个名字里带齐全部参数的 GPU kernel。
3. **`via P2P/IPC` 是什么**：本篇只读到"NCCL 给机内的边选了显存直写"这句话本身——它是什么动作，下一篇亲手做一遍。

你现在拥有的：一套语义模型（第 2 篇），加上调用下潜每一站的亲眼证据（本篇）。下一篇到站底第一层硬件。

**练习题**（按难度排序）：

1. 把 `exp3` 的 tensor 从 4 MB 改成 4 KB、400 MB 各跑一次：第一次还慢吗？后面几次的时间怎么变？验证"建组成本与消息大小无关"。
2. 在 `exp4` 里把 tensor 从 4 KB 到 1 GB 扫几个量级，看 profiler 里 kernel 名的协议后缀（LL / LL128 / Simple）怎么切换，记录切换点——NCCL 深挖篇目讲协议选择时对答案。
3. 用 4、8 进程跑第六节的日志命令，把 `Ring 00` 的走向画出来，和 `nvidia-smi topo -m` 的矩阵对照：环的形状和你机器的 NVLink 拓扑有什么关系？
4. 思考题：如果改成 `init_process_group` 时就建好所有 communicator（不懒初始化），会发生什么？提示：Megatron 启动时要建几十个子组，想想启动时间和显存占用。

---

## 附录 A：报错锦囊

| 症状                                | 原因与处置                                                   |
| ----------------------------------- | ------------------------------------------------------------ |
| 计时全是个位数 μs，连第一次也"不慢" | 忘了 synchronize：`all_reduce` 是异步派发，你测的是 CPU 返回时间。计时前后各加一次 `torch.cuda.synchronize()` |
| profiler 表格没有 CUDA 列           | `activities` 少加了 `ProfilerActivity.CUDA`；或者你在 `--device cpu` 下跑的 |
| `NCCL_DEBUG` 没有任何输出           | 日志走 stderr，管道前加 `2>&1`；或者环境变量没传进子进程     |
| kernel 名和书上对不上               | torch/NCCL 版本差异，字段顺序会变；认出"算法 / 协议 / 规约 / dtype"四类字段即可 |
| 第一次 `all_reduce` 挂住不动        | 建组也是集合操作，要全员到场；有人没到就先往上翻第一个 traceback |

## 附录 B：本篇文件清单

```javascript
数据盘/tp-lab/
├── mini_tp.py           # 第 2 篇：lab_init 双模式入口
├── exp0_hello.py        # 第 2 篇：环境自检
├── exp1_groups.py       # 第 2 篇：子组实验
├── exp2_primitives.py   # 第 2 篇：原语家族
├── exp3_timing.py       # 本篇：第一次 all_reduce 为什么慢
└── exp4_profiler.py     # 本篇：profiler 看 all_reduce 的真身（GPU 专属）
```

exp3 两种模式都能跑（gloo 的第一次同样慢）；exp4 只在 GPU/NCCL 模式下有意义。运行方式照旧：`torchrun --nproc_per_node=N 脚本名.py`。

## 下篇预告

《机内通路：NVLink 上的一次显存直写》——把 `via P2P/IPC` 拆到底：不开 NCCL，亲手写一个 kernel，让一条 store 指令穿过 NVLink 写进另一块卡的显存。