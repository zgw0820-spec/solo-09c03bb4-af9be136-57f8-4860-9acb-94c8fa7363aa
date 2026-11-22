"""单个 Gloo 训练 rank 的演练进程。

由 Flask 后端以子进程方式启动两个实例（rank 0 / rank 1），通过 env:// 方式
在本机回环网络上组成一个 2 节点的 Gloo 进程组，同步一个小线性模型的梯度。

所有关键进度都以 JSON Lines 追加写入事件文件，Flask 侧只读事件文件来还原
“实际进入 / 完成了哪些同步步”，不依赖进程退出码。

用法:
    python trainer.py --run-dir DIR --rank 0 --batches 5
                      [--pause-before-step 3]
其余分布式参数通过环境变量传入:
    MASTER_ADDR / MASTER_PORT / WORLD_SIZE / RANK
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gloo 梯度同步演练 rank")
    parser.add_argument("--run-dir", required=True, help="本轮演练目录")
    parser.add_argument("--rank", type=int, required=True, choices=(0, 1))
    parser.add_argument("--batches", type=int, required=True, help="本 rank 的样本批数（训练步数）")
    parser.add_argument(
        "--pause-before-step",
        type=int,
        default=None,
        help="若给出，则在第 N 步进入同步之前暂停，等待 resume 文件后继续",
    )
    return parser.parse_args()


class EventLog:
    """JSONL 事件日志，每次写入都 flush，保证后端能实时读到最新进度。"""

    def __init__(self, path: str, rank: int) -> None:
        self._fp = open(path, "a", encoding="utf-8")
        self.rank = rank

    def emit(self, event: str, **fields) -> None:
        record = {"ts": time.time(), "rank": self.rank, "event": event}
        record.update(fields)
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()
        os.fsync(self._fp.fileno())

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    rank = args.rank

    stop_flag = os.path.join(run_dir, "stop")
    resume_flag = os.path.join(run_dir, f"resume.rank{rank}")
    events = EventLog(os.path.join(run_dir, f"rank{rank}.events.jsonl"), rank)

    # SIGTERM/SIGINT 时尽量走干净的退出路径；若正阻塞在集合通信里，
    # Python 信号处理会在调用被中断后抛出 SystemExit。
    def _handle_term(signum, _frame):
        events.emit("signal", signal=signal.Signals(signum).name)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    if os.path.exists(stop_flag):
        events.emit("aborted_before_start", reason="stop flag present")
        events.close()
        return 0

    events.emit(
        "start",
        pid=os.getpid(),
        batches=args.batches,
        pause_before_step=args.pause_before_step,
        master_port=os.environ.get("MASTER_PORT"),
    )

    # 小模型：4 维输入 -> 2 维输出的线性层，CPU 上即可运行
    torch.manual_seed(20261122 + rank)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    try:
        # 给足 1 小时超时：演练的“卡住”由后端依据事件流判断，
        # 不希望 Gloo 自己先超时把证据打断。
        dist.init_process_group(
            backend="gloo",
            timeout=datetime.timedelta(hours=1),
        )
        events.emit("init_done", backend="gloo", world_size=dist.get_world_size())

        for step in range(1, args.batches + 1):
            if os.path.exists(stop_flag):
                events.emit("stopped", step=step)
                break

            # 暂停闸门：在指定步“进入同步之前”停住，模拟一侧分片处理被挂起
            if args.pause_before_step is not None and step == args.pause_before_step:
                events.emit("paused", step=step)
                while not os.path.exists(resume_flag):
                    if os.path.exists(stop_flag):
                        events.emit("stopped", step=step, while_paused=True)
                        events.close()
                        return 0
                    time.sleep(0.2)
                events.emit("resumed", step=step)

            # 造一批本地样本并前向 + 反向
            batch_size = 8
            x = torch.randn(batch_size, 4)
            target = torch.randn(batch_size, 2)
            optimizer.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()

            # 进入梯度同步（all-reduce 求平均），这是本演练要观测的关键点
            events.emit("sync_enter", step=step, loss=round(float(loss.detach()), 6))
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()
            dist.barrier()
            events.emit("sync_done", step=step)

            optimizer.step()
            events.emit("step_done", step=step)
        else:
            events.emit("training_complete", steps=args.batches)

        if dist.is_initialized():
            dist.destroy_process_group()
        events.emit("exit", code=0)
        return 0

    except SystemExit:
        # 信号触发的退出：尽量清理进程组，再把退出事件落盘
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        events.emit("exit", code=143, reason="terminated")
        return 143
    except Exception as exc:  # noqa: BLE001 - 演练需要把任何失败写成证据
        events.emit(
            "sync_error",
            error_type=type(exc).__name__,
            error=str(exc)[:500],
        )
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        events.emit("exit", code=1, reason="error")
        return 1
    finally:
        events.close()


if __name__ == "__main__":
    sys.exit(main())
