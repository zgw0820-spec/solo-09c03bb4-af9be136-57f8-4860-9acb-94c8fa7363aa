"""梯度同步卡住演练 —— Flask 排障后端。

职责：
1. 接收训练员填写的两侧样本批数、可选暂停设置，启动两个 Gloo CPU rank 子进程；
2. 看门狗线程只读各 rank 的 JSONL 事件文件，还原“实际进入 / 完成的同步步”；
3. 满足演练结束条件（一侧跑完、另一侧已进入下一次同步）时给出卡点结论；
   超时但证据不足时标记“原因待查”；
4. 任何结束路径都先回收两个训练进程，读完本轮已落盘的进度后再**一次性**发布终态；
   终态只发布一次，停服与看门狗同时收尾也不会互相覆盖；
5. 服务停止（docker stop / SIGTERM）时保留看门狗已作出的结论，仍在运行的演练
   标记为“服务中断”；重启后仍可用原链接查回同一份结果。
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# 兼容 `python -m app.server`（容器）和 `python app/server.py`（本地自测）两种启动方式
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.environ.get("DRILL_BASE_DIR", APP_DIR.parent / "runs"))
PYTHON_BIN = os.environ.get("DRILL_PYTHON", sys.executable)
# 容器内依赖装在系统 site-packages，可用 -I 隔离环境；本地自测时可置空。
PYTHON_FLAGS = os.environ.get("DRILL_PYTHON_FLAGS", "-I").split()
POLL_INTERVAL = 0.25  # 秒
REAP_GRACE = 3  # SIGTERM 后等待多久升级为 SIGKILL
REAP_KILL_WAIT = 2  # SIGKILL 后最多再等多久
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

INTERRUPTED_CONCLUSION = {
    "title": "服务停止，演练中断",
    "detail": (
        "服务在演练给出结论前收到停止信号；两个训练进程已回收，本轮没有形成有效结论。"
        "下面的进度来自进程退出前已落盘的事件日志。"
    ),
    "interrupted": True,
}
RECOVERED_CONCLUSION = {
    "title": "服务中断，未找到终态记录",
    "detail": (
        "服务可能在发布结论前被强制终止；本轮没有终态记录。"
        "下面的进度由事件文件还原，进程回收结果未记录。"
    ),
    "interrupted": True,
    "recovered": True,
}

app = Flask(__name__)
# 可重入：SIGTERM 处理器在主线程里取锁，可能正好打断主线程已持锁的请求处理
_lock = threading.RLock()


@dataclass
class RunHandle:
    run_id: str
    run_dir: Path
    params: dict
    start_ts: float
    timeout_s: int
    procs: dict[int, subprocess.Popen]
    stop_requested: bool = False
    finalized: bool = False
    finalizing: bool = False
    thread: threading.Thread | None = None
    final_status: dict | None = None
    pending_verdict: tuple[str, dict] | None = None
    # 看门狗与停服收尾共用，保证终态只发布一次
    finalize_lock: threading.RLock = field(default_factory=threading.RLock)


_current: RunHandle | None = None
_shutting_down = False


# ---------------------------------------------------------------- 事件读取

def read_events(run_dir: Path, rank: int) -> list[dict]:
    path = run_dir / f"rank{rank}.events.jsonl"
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def rank_summary(events: list[dict], alive: bool) -> dict:
    entered: list[int] = []
    completed: list[int] = []
    last_event = events[-1] if events else None
    paused_at: int | None = None
    error: dict | None = None
    training_complete = False
    exit_code: int | None = None

    for ev in events:
        name = ev.get("event")
        step = ev.get("step")
        if name == "sync_enter" and isinstance(step, int):
            entered.append(step)
        elif name == "sync_done" and isinstance(step, int):
            completed.append(step)
        elif name == "paused":
            paused_at = step
        elif name == "sync_error":
            error = {"type": ev.get("error_type"), "message": ev.get("error", "")}
        elif name == "training_complete":
            training_complete = True
        elif name == "exit":
            exit_code = ev.get("code")

    last_name = last_event.get("event") if last_event else None
    if training_complete:
        state = "已完成全部批次"
    elif error:
        state = f"同步报错（{error['type']}）"
    elif last_name == "paused":
        state = f"暂停中（第 {paused_at} 步同步前）"
    elif last_name == "sync_enter":
        if alive:
            state = f"阻塞在第 {entered[-1]} 步梯度同步"
        else:
            state = f"进程退出前阻塞在第 {entered[-1]} 步梯度同步（code={exit_code}）"
    elif not alive:
        state = f"已退出（code={exit_code}）"
    elif not events:
        state = "等待启动"
    else:
        state = "运行中"

    return {
        "alive": alive,
        "entered": sorted(set(entered)),
        "completed": sorted(set(completed)),
        "last_event": last_event,
        "state": state,
        "paused_at": paused_at,
        "error": error,
        "training_complete": training_complete,
        "exit_code": exit_code,
    }


def _live_status(handle: RunHandle) -> dict:
    """读取当前进程与事件文件，构造“运行中”快照；绝不返回终态。"""
    ranks: dict[str, dict] = {}
    for rank in (0, 1):
        proc = handle.procs[rank]
        return_code = proc.poll()
        alive = return_code is None
        summary = rank_summary(read_events(handle.run_dir, rank), alive)
        if return_code is not None and summary["exit_code"] is None:
            summary["exit_code"] = return_code
        summary["pid"] = proc.pid
        ranks[str(rank)] = summary

    return {
        "run_id": handle.run_id,
        "status": "running",
        "created_at": handle.start_ts,
        "now": time.time(),
        "timeout_s": handle.timeout_s,
        "params": handle.params,
        "ranks": ranks,
        "conclusion": None,
        "stop_requested": handle.stop_requested,
        "finalized": False,
        "finalizing": handle.finalizing,
    }


def build_status(handle: RunHandle) -> dict:
    # 终态一旦发布就是不可变的：网页首次看到的终态与 status.json / 重启后回查完全一致
    if handle.final_status is not None:
        return handle.final_status
    return _live_status(handle)


# ---------------------------------------------------------------- 终态发布

def _atomic_write_json(path: Path, doc: dict) -> None:
    """先写临时文件并 fsync，再原子替换，避免读到半份 status.json。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(doc, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def reap(handle: RunHandle) -> dict:
    """SIGTERM -> 等待 -> 必要时 SIGKILL，确保两个训练进程都退出后才返回。"""
    for proc in handle.procs.values():
        if proc.poll() is None:
            try:
                proc.terminate()
            except (ProcessLookupError, OSError):
                pass

    deadline = time.time() + REAP_GRACE
    reaped: dict[str, dict] = {}
    for rank, proc in sorted(handle.procs.items()):
        code = proc.poll()
        if code is None:
            try:
                code = proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    code = proc.wait(timeout=REAP_KILL_WAIT)
                except subprocess.TimeoutExpired:
                    code = proc.poll()
        reaped[str(rank)] = {"pid": proc.pid, "exit_code": code}
    return reaped


def finalize(handle: RunHandle, verdict: str | None, conclusion: dict | None,
             reason: str = "watchdog") -> dict:
    """回收两个 rank 后原子地发布唯一一份终态。

    - 看门狗传入已经判定好的 verdict/conclusion；
    - 停服路径传入 None：优先沿用看门狗已经作出的 pending 结论，否则标记服务中断；
    - 回收动作本身可能让对端 Gloo 报连接关闭，但这些事件只进入 ranks 快照，
      不会反过来改写已经定下的结论。
    """
    global _current
    with handle.finalize_lock:
        if handle.final_status is not None:
            return handle.final_status

        source = reason
        if verdict is None:
            if handle.pending_verdict is not None:
                verdict, conclusion = handle.pending_verdict
                source = "shutdown"
            elif handle.stop_requested:
                # 网页已请求停止，只是服务先收到了 SIGTERM：尊重停止意图
                verdict = "stopped"
                conclusion = {
                    "title": "演练被手动停止，服务随后退出；两个训练进程已回收",
                    "detail": "停止指令已下发，两个训练进程均被终止。",
                }
                source = "shutdown"
            else:
                verdict = "interrupted"
                conclusion = dict(INTERRUPTED_CONCLUSION)

        handle.finalizing = True
        try:
            reaped = reap(handle)
            # 回收完成后再读一次事件：拿到进程退出前已落盘的进度与退出/报错事件
            final = _live_status(handle)
            final["status"] = verdict
            final["conclusion"] = conclusion
            final["reaped"] = reaped
            final["finished_at"] = time.time()
            final["finalized"] = True
            final["finalizing"] = False
            final["finalized_by"] = source

            try:
                _atomic_write_json(handle.run_dir / "status.json", final)
            except OSError as exc:
                # 磁盘异常时仍把结果留在内存里供本次进程回查，并暴露发布失败
                final["publish_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[finalize] 写入 status.json 失败: {exc}", file=sys.stderr)
            # 文件落盘之后才在内存里标记终态，确保任何“终态”响应都带完整回收记录
            handle.final_status = final
        finally:
            handle.finalizing = False

        handle.finalized = True
        with _lock:
            if _current is handle:
                _current = None
        return final


# ---------------------------------------------------------------- 结论判定

def classify(handle: RunHandle, ranks: dict[str, dict]) -> tuple[str | None, dict | None]:
    """返回 (status, conclusion)；status 为 None 表示继续观察。"""
    r0, r1 = ranks["0"], ranks["1"]
    any_alive = r0["alive"] or r1["alive"]

    # 1) 两侧都跑完：正常结束
    if r0["training_complete"] and r1["training_complete"]:
        return "completed", {
            "title": "两侧 rank 均正常跑完，梯度同步未出现卡点",
            "detail": (
                f"rank0 完成 {len(r0['completed'])} 步、"
                f"rank1 完成 {len(r1['completed'])} 步，所有同步步均有进有出。"
            ),
        }

    finished_rank: int | None = None
    other_rank: int | None = None
    if r0["training_complete"] and not r1["training_complete"]:
        finished_rank, other_rank = 0, 1
    elif r1["training_complete"] and not r0["training_complete"]:
        finished_rank, other_rank = 1, 0

    # 2) 一侧跑完，另一侧已进入它没到过的下一步同步 —— 演练目标场景
    if finished_rank is not None:
        fin = ranks[str(finished_rank)]
        oth = ranks[str(other_rank)]
        fin_max = max(fin["entered"], default=0)
        oth_entered = oth["entered"]
        if oth_entered and max(oth_entered) > fin_max:
            blocked_step = max(oth_entered)
            # 证据 A：另一侧仍活着且最后动作就是进入同步（当场抓到阻塞）
            live_blocked = (
                oth["alive"]
                and oth["last_event"] is not None
                and oth["last_event"].get("event") == "sync_enter"
                and oth["last_event"].get("step") == blocked_step
            )
            # 证据 B：对端退出后 Gloo 报连接关闭，事件流仍能还原卡点
            peer_closed = (
                oth["error"] is not None
                and not oth["alive"]
                and blocked_step not in oth["completed"]
            )
            if live_blocked or peer_closed:
                evidence = (
                    "当场观测：该 rank 仍存活，最后事件为 sync_enter 且无对应 sync_done"
                    if live_blocked
                    else "事后还原：该 rank 进入同步后对端拆除，Gloo 报连接关闭，sync_enter 无对应 sync_done"
                )
                return "stuck", {
                    "title": (
                        f"确认卡点：rank{other_rank} 阻塞在第 {blocked_step} 步梯度同步，"
                        f"rank{finished_rank} 已跑完 {fin_max} 步退出"
                    ),
                    "stuck_rank": other_rank,
                    "finished_rank": finished_rank,
                    "blocked_step": blocked_step,
                    "finished_steps": fin_max,
                    "evidence_type": "observed_block" if live_blocked else "peer_closed",
                    "detail": (
                        f"rank{finished_rank} 只有 {fin_max} 个批次，跑完后销毁 Gloo 进程组退出；"
                        f"rank{other_rank} 仍有批次，已进入第 {blocked_step} 步 all_reduce，"
                        f"永远等不到对端。{evidence}。"
                    ),
                    "error": oth["error"],
                }

    # 3) 两侧都死了但既没都完成、也不满足卡点证据：异常
    if not any_alive:
        return "error", {
            "title": "两个训练进程均已退出，但未形成有效的卡点证据",
            "detail": (
                f"rank0 状态：{r0['state']}；rank1 状态：{r1['state']}。"
                "请查看各 rank 的 stderr 事件日志排查初始化或运行时错误。"
            ),
        }

    # 4) 超时但证据不足：原因待查
    elapsed = time.time() - handle.start_ts
    if elapsed >= handle.timeout_s:
        observations = []
        for rank in (0, 1):
            s = ranks[str(rank)]
            observations.append(
                f"rank{rank}：进入同步步 {s['entered'] or '无'}，"
                f"完成同步步 {s['completed'] or '无'}，当前状态「{s['state']}」"
            )
        return "timeout", {
            "title": "演练超时，证据不足，原因待查",
            "reason_pending": True,
            "detail": (
                f"达到 {handle.timeout_s}s 超时仍未出现“一侧跑完、另一侧进入下一次同步”的判定条件。"
                "已有观测：" + "；".join(observations) + "。需结合分片配置与暂停设置进一步排查。"
            ),
        }

    return None, None


def watchdog(handle: RunHandle) -> None:
    global _current
    try:
        while True:
            time.sleep(POLL_INTERVAL)
            if handle.final_status is not None:
                return
            try:
                # 先读事件再判定，保证结论基于最新落盘进度
                ranks = _live_status(handle)["ranks"]
                if handle.stop_requested:
                    verdict, conclusion = "stopped", {
                        "title": "演练被手动停止，进程已回收",
                        "detail": "停止指令已下发，两个训练进程（若仍存活）均被终止。",
                    }
                else:
                    verdict, conclusion = classify(handle, ranks)
            except Exception as exc:  # noqa: BLE001 - 后端异常也要留下可回查的终态
                finalize(handle, "error", {
                    "title": "后端判定异常，演练已终止",
                    "detail": f"看门狗读取事件或判定时出错：{type(exc).__name__}: {exc}",
                })
                return

            if verdict is None:
                continue

            # 先记下结论：停服信号若在此刻插入，收尾会沿用这份结论而不是改成“服务中断”
            handle.pending_verdict = (verdict, conclusion)
            finalize(handle, verdict, conclusion, reason="watchdog")
            return
    except Exception as exc:  # noqa: BLE001 - 兜底，避免 _current 永远卡住
        print(f"[watchdog] 未捕获异常: {exc}", file=sys.stderr)
        try:
            finalize(handle, "error", {
                "title": "看门狗异常退出，演练已终止",
                "detail": f"{type(exc).__name__}: {exc}",
            })
        except Exception as final_exc:  # noqa: BLE001
            print(f"[watchdog] 收尾失败: {final_exc}", file=sys.stderr)
            with _lock:
                if _current is handle:
                    _current = None


# ---------------------------------------------------------------- 启动演练

def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_run(params: dict) -> RunHandle:
    global _current
    with _lock:
        if _current is not None and _current.final_status is None:
            raise RuntimeError("已有演练在运行，请等待结束或先停止后再重跑")

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_dir = BASE_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(run_dir / "params.json", params)

        port = _free_port()
        procs: dict[int, subprocess.Popen] = {}
        for rank in (0, 1):
            env = os.environ.copy()
            env.update(
                MASTER_ADDR="127.0.0.1",
                MASTER_PORT=str(port),
                WORLD_SIZE="2",
                RANK=str(rank),
                OMP_NUM_THREADS="1",
            )
            cmd = [
                PYTHON_BIN,
                *PYTHON_FLAGS,
                str(APP_DIR / "trainer.py"),
                "--run-dir",
                str(run_dir),
                "--rank",
                str(rank),
                "--batches",
                str(params["batches"][rank]),
            ]
            if params["pause_rank"] == rank and params["pause_before_step"]:
                cmd += ["--pause-before-step", str(params["pause_before_step"])]
            out = open(run_dir / f"rank{rank}.out", "a", encoding="utf-8")
            err = open(run_dir / f"rank{rank}.err", "a", encoding="utf-8")
            try:
                procs[rank] = subprocess.Popen(
                    cmd, env=env, stdout=out, stderr=err, cwd=str(APP_DIR)
                )
            finally:
                # 子进程已继承自己的 fd，父进程侧关闭，避免长跑服务累积句柄
                out.close()
                err.close()

        handle = RunHandle(
            run_id=run_id,
            run_dir=run_dir,
            params=params,
            start_ts=time.time(),
            timeout_s=int(params["timeout_s"]),
            procs=procs,
        )
        # 先登记再启动看门狗，避免看门狗在 _current 赋值前就完成收尾
        _current = handle
        thread = threading.Thread(target=watchdog, args=(handle,), daemon=True)
        handle.thread = thread
        thread.start()
        return handle


# ---------------------------------------------------------------- 回查历史

def _synthesize_interrupted(run_id: str, run_dir: Path) -> dict | None:
    """没有 status.json 时，用已落盘的事件还原一份“服务中断”结果。

    覆盖服务被 SIGKILL 等无法执行收尾逻辑的情况，避免历史演练链接 404。
    结果只依赖磁盘内容，重复回查/重启后读取完全一致。
    """
    params_path = run_dir / "params.json"
    params: dict | None = None
    if params_path.exists():
        try:
            loaded = json.loads(params_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                params = loaded
        except (OSError, json.JSONDecodeError):
            params = None
    if params is None and not any(
        (run_dir / f"rank{rank}.events.jsonl").exists() for rank in (0, 1)
    ):
        # 空目录（刚建好还没写参数/事件）不当作演练，避免首页出现空链接
        return None
    params = params or {
        "batches": [None, None],
        "pause_rank": None,
        "pause_before_step": None,
        "timeout_s": None,
    }

    ranks: dict[str, dict] = {}
    timestamps: list[float] = []
    for rank in (0, 1):
        events = read_events(run_dir, rank)
        summary = rank_summary(events, alive=False)
        summary["pid"] = None
        ranks[str(rank)] = summary
        timestamps.extend(
            ev["ts"] for ev in events if isinstance(ev.get("ts"), (int, float))
        )

    try:
        mtime = run_dir.stat().st_mtime
    except OSError:
        mtime = time.time()
    created_at = min(timestamps) if timestamps else mtime
    finished_at = max(timestamps) if timestamps else mtime

    return {
        "run_id": run_id,
        "status": "interrupted",
        "created_at": created_at,
        "now": finished_at,
        "timeout_s": params.get("timeout_s"),
        "params": params,
        "ranks": ranks,
        "conclusion": dict(RECOVERED_CONCLUSION),
        "stop_requested": (run_dir / "stop").exists(),
        "finalized": True,
        "finalizing": False,
        "reaped": None,
        "finished_at": finished_at,
        "finalized_by": "recovery",
        "recovered": True,
    }


def load_stored_run(run_id: str) -> dict | None:
    if not RUN_ID_RE.match(run_id) or run_id in (".", ".."):
        return None
    run_dir = BASE_DIR / run_id
    if not run_dir.is_dir():
        return None

    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 半份文件（旧版本非原子写入 / 磁盘异常）也走事件还原，避免 404
            pass
    return _synthesize_interrupted(run_id, run_dir)


# ---------------------------------------------------------------- 路由

@app.get("/")
def index():
    runs = sorted(
        (p for p in BASE_DIR.glob("*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    recent = []
    for path in runs[:10]:
        if (
            _current is not None
            and _current.run_id == path.name
            and _current.final_status is None
        ):
            recent.append({"run_id": path.name, "status": "running"})
            continue
        stored = load_stored_run(path.name)
        recent.append({
            "run_id": path.name,
            "status": stored["status"] if stored else "unknown",
        })
    return render_template(
        "index.html", recent=recent, current=_current,
        status_text={
            "running": "运行中", "stuck": "已确认卡点", "completed": "正常跑完",
            "timeout": "超时 · 原因待查", "error": "异常退出",
            "stopped": "已手动停止", "interrupted": "服务中断",
            "unknown": "状态未知",
        },
    )


@app.get("/runs/<run_id>")
def run_page(run_id):
    return render_template("run.html", run_id=run_id)


@app.post("/api/runs")
def api_start():
    form = request.form
    try:
        batches = [int(form["batches0"]), int(form["batches1"])]
        if any(b < 1 or b > 1000 for b in batches):
            raise ValueError("每侧批数需在 1~1000 之间")
        pause_rank_raw = form.get("pause_rank", "")
        pause_rank = int(pause_rank_raw) if pause_rank_raw != "" else None
        if pause_rank is not None and pause_rank not in (0, 1):
            raise ValueError("暂停侧只能是 rank0 或 rank1")
        pause_before_step = int(form.get("pause_before_step") or 0) or None
        if pause_rank is not None:
            if not pause_before_step:
                raise ValueError("选择暂停侧时必须指定暂停步")
            if not (1 <= pause_before_step <= batches[pause_rank]):
                raise ValueError("暂停步超出该侧批数范围")
        timeout_s = int(form.get("timeout_s", 30))
        if not (3 <= timeout_s <= 600):
            raise ValueError("超时时间需在 3~600 秒之间")
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"参数无效：{exc}"}), 400

    params = {
        "batches": batches,
        "pause_rank": pause_rank,
        "pause_before_step": pause_before_step,
        "timeout_s": timeout_s,
    }
    try:
        handle = start_run(params)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"run_id": handle.run_id})


@app.get("/api/runs")
def api_list():
    runs = sorted(
        (p for p in BASE_DIR.glob("*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return jsonify({"runs": [p.name for p in runs[:20]]})


@app.get("/api/runs/<run_id>")
def api_status(run_id):
    with _lock:
        current = _current
    if current is not None and current.run_id == run_id:
        # 收尾期间返回实时进度（含 finalizing 标记）；终态返回已落盘的那一份
        return jsonify(build_status(current))
    stored = load_stored_run(run_id)
    if stored is None:
        return jsonify({"error": "演练不存在"}), 404
    return jsonify(stored)


@app.post("/api/runs/<run_id>/resume")
def api_resume(run_id):
    with _lock:
        handle = _current
    if handle is None or handle.run_id != run_id:
        return jsonify({"error": "该演练不在运行中，无法恢复"}), 409
    if handle.final_status is not None or handle.finalizing:
        return jsonify({"error": "演练正在收尾或已结束，无法恢复"}), 409
    rank = handle.params.get("pause_rank")
    if rank is None:
        return jsonify({"error": "本轮没有配置暂停侧"}), 400
    try:
        (handle.run_dir / f"resume.rank{rank}").touch()
    except OSError as exc:
        return jsonify({"error": f"无法写入恢复标记：{exc}"}), 500
    return jsonify({"ok": True, "resumed_rank": rank})


@app.post("/api/runs/<run_id>/stop")
def api_stop(run_id):
    with _lock:
        handle = _current
    if handle is None or handle.run_id != run_id:
        return jsonify({"error": "该演练不在运行中"}), 409
    if handle.final_status is not None:
        return jsonify({"error": "演练已结束"}), 409
    handle.stop_requested = True
    if not handle.finalizing:
        try:
            (handle.run_dir / "stop").touch()
        except OSError as exc:
            return jsonify({"error": f"无法写入停止标记：{exc}"}), 500
    return jsonify({"ok": True})


def _shutdown_cleanup(*_args) -> None:
    """服务退出前的统一收尾：保留看门狗结论，或把在跑演练标记为服务中断。"""
    with _lock:
        handle = _current
    if handle is None:
        return
    try:
        (handle.run_dir / "stop").touch()
    except OSError:
        pass
    finalize(handle, None, None, reason="shutdown")


import atexit  # noqa: E402


def _handle_term(signum, _frame) -> None:
    """SIGTERM/SIGINT：先发布唯一一份终态并回收两个训练进程，再退出。

    SIGTERM 默认动作会直接终止进程，atexit 不会执行，所以必须显式处理。
    """
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    # 收尾期间忽略后续 TERM/INT，避免重入 finalize 或提前打断回收
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _shutdown_cleanup()
    raise SystemExit(0)


# 正常解释器退出（含上面处理器之后的退出）走 atexit 兜底
atexit.register(_shutdown_cleanup)
signal.signal(signal.SIGTERM, _handle_term)
signal.signal(signal.SIGINT, _handle_term)


if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), use_reloader=False)
