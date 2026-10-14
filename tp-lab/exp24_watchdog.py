# exp24_watchdog.py —— 实验四：亲手挂死一次 all_reduce，看看门狗怎么抓人
# 本篇唯一不用 lab_init 的实验：要把默认 10 分钟的超时调短，自己初始化。
import datetime
import os
import sys
import time

import torch
import torch.distributed as dist


def main():
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else (
        "cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if device == "cuda" else "gloo"
    # timeout=30s：把第 2 篇那笔"默认 10 分钟"的账调到肉眼可等的长度
    dist.init_process_group(backend, timeout=datetime.timedelta(seconds=30))
    rank = dist.get_rank()
    if device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cpu")

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] rank{rank}：{msg}", flush=True)

    t = torch.ones(1024, device=dev)
    dist.all_reduce(t)          # 第 1 号 collective：全员到场，正常（建组在此完成）
    if dev.type == "cuda":
        torch.cuda.synchronize()
    log("第 1 次 all_reduce 完成，组建好了")

    if rank == 0:
        # 模拟 bug：rank 0 走了别的分支，第 2 号 collective 不参加了
        log("我这边出了 bug，第 2 次 all_reduce 不参加了（假装还在干活）")
        for i in range(40):
            time.sleep(1)
            if i % 10 == 0:
                log(f"还在假装干活…（第 {i} 秒）")
    else:
        log("发起第 2 次 all_reduce，等 rank0…")
        h = dist.all_reduce(t, async_op=True)   # 第 2 号 collective：永远缺一个人
        log("调用已经返回（异步派发！）——真正的卡在 wait/同步里")
        h.wait(timeout=datetime.timedelta(seconds=60))  # CPU 真等：60s 兜底
        log("这行不会被执行")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
