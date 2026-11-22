"""梯度同步卡住演练 —— Flask 排障后端。

职责：
1. 接收训练员填写的两侧样本批数、可选暂停设置，启动两个 Gloo CPU rank 子进程；
2. 看门狗线程只读各 rank 的 JSONL 事件文件，还原“实际进入 / 完成的同步步”；
3. 满足演练结束条件（一侧跑完、另一侧已进入下一次同步）时给出卡点结论；
   超时但证据不足时标记“原因待查”；
4. 任何结束路径都回收两个训练进程，读完本轮已落盘的进度后再下结论；
5. 结束后允许用相同或新参数重跑。
"""
from __future__ import annotations

import json
import os
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
REAP_GRACE = 5  # SIGTERM 后等待多久升级为 SIGKILL

app = Flask(__name__)
_lock = threading.Lock()


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
    thread: threading.Thread | None = None
    final_status: dict | None = None


_current: RunHandle | None = None


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

    if training_complete:
        state = "已完成全部批次"
    elif error:
        state = f"同步报错（{error['type']}）"
    elif last_event and last_event.get("event") == "paused":
        state = f"暂停中（第 {paused_at} 步同步前）"
    elif last_event and last_event.get("event") == "sync_enter":
        state = f"阻塞在第 {entered[-1]} 步梯度同步"
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


def build_status(handle: RunHandle) -> dict:
    run_dir = handle.run_dir
    ranks: dict[str, dict] = {}
    for rank in (0, 1):
        proc = handle.procs[rank]
        alive = proc.poll() is None
        summary = rank_summary(read_events(run_dir, rank), alive)
        summary["pid"] = proc.pid
        ranks[str(rank)] = summary

    stored = {}
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            stored = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stored = {}

    return {
        "run_id": handle.run_id,
        "status": stored.get("status", "running"),
        "created_at": handle.start_ts,
        "now": time.time(),
        "timeout_s": handle.timeout_s,
        "params": handle.params,
        "ranks": ranks,
        "conclusion": stored.get("conclusion"),
        "stop_requested": handle.stop_requested,
        "finalized": handle.finalized,
    }


# ---------------------------------------------------------------- 进程回收

def reap(handle: RunHandle) -> dict:
    """SIGTERM -> 等待 -> 必要时 SIGKILL，确保两个训练进程都被回收。"""
    reaped: dict[str, dict] = {}
    for rank, proc in handle.procs.items():
        if proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
    deadline = time.time() + REAP_GRACE
    for rank, proc in handle.procs.items():
        remain = max(0.1, deadline - time.time())
        try:
            code = proc.wait(timeout=remain)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait(timeout=5)
        reaped[str(rank)] = {"pid": proc.pid, "exit_code": code}
    return reaped


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
    while True:
        time.sleep(POLL_INTERVAL)
        # 先读事件再判定，保证结论基于最新落盘进度
        status = build_status(handle)
        ranks = status["ranks"]

        if handle.stop_requested:
            verdict, conclusion = "stopped", {
                "title": "演练被手动停止，进程已回收",
                "detail": "停止指令已下发，两个训练进程（若仍存活）均被终止。",
            }
        else:
            verdict, conclusion = classify(handle, ranks)

        if verdict is None:
            continue

        reaped = reap(handle)
        # 回收后再读一次事件，确保退出/报错事件都已落盘后再写结论
        final = build_status(handle)
        final["status"] = verdict
        final["conclusion"] = conclusion
        final["reaped"] = reaped
        final["finished_at"] = time.time()
        (handle.run_dir / "status.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        handle.finalized = True
        handle.final_status = final
        with _lock:
            if _current is handle:
                _current = None
        return


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
        if _current is not None and not _current.finalized:
            raise RuntimeError("已有演练在运行，请等待结束或先停止后再重跑")

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_dir = BASE_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
        )

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
            procs[rank] = subprocess.Popen(
                cmd, env=env, stdout=out, stderr=err, cwd=str(APP_DIR)
            )

        handle = RunHandle(
            run_id=run_id,
            run_dir=run_dir,
            params=params,
            start_ts=time.time(),
            timeout_s=int(params["timeout_s"]),
            procs=procs,
        )
        thread = threading.Thread(target=watchdog, args=(handle,), daemon=True)
        handle.thread = thread
        thread.start()
        _current = handle
        return handle


def load_stored_run(run_id: str) -> dict | None:
    run_dir = BASE_DIR / run_id
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return None
    return json.loads(status_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 路由

@app.get("/")
def index():
    runs = sorted(
        (p for p in BASE_DIR.glob("*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    recent = [p.name for p in runs[:10]]
    return render_template("index.html", recent=recent, current=_current)


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
        if _current is not None and _current.run_id == run_id:
            return jsonify(build_status(_current))
    stored = load_stored_run(run_id)
    if stored is None:
        return jsonify({"error": "演练不存在"}), 404
    return jsonify(stored)


@app.post("/api/runs/<run_id>/resume")
def api_resume(run_id):
    with _lock:
        if _current is None or _current.run_id != run_id:
            return jsonify({"error": "该演练不在运行中，无法恢复"}), 409
        handle = _current
        rank = handle.params.get("pause_rank")
        if rank is None:
            return jsonify({"error": "本轮没有配置暂停侧"}), 400
        flag = handle.run_dir / f"resume.rank{rank}"
        flag.touch()
    return jsonify({"ok": True, "resumed_rank": rank})


@app.post("/api/runs/<run_id>/stop")
def api_stop(run_id):
    with _lock:
        if _current is None or _current.run_id != run_id:
            return jsonify({"error": "该演练不在运行中"}), 409
        handle = _current
        handle.stop_requested = True
        (handle.run_dir / "stop").touch()
    return jsonify({"ok": True})


def _shutdown_cleanup(*_args) -> None:
    with _lock:
        handle = _current
    if handle is not None and not handle.finalized:
        (handle.run_dir / "stop").touch()
        reap(handle)


import atexit  # noqa: E402


def _handle_term(signum, _frame) -> None:
    """SIGTERM/SIGINT：先回收两个训练进程，再正常退出（atexit 不会在
    SIGTERM 默认动作下执行，所以必须显式处理）。"""
    _shutdown_cleanup()
    raise SystemExit(0)


# 正常解释器退出（含上面处理器之后的退出）走 atexit 兜底
atexit.register(_shutdown_cleanup)
signal.signal(signal.SIGTERM, _handle_term)
signal.signal(signal.SIGINT, _handle_term)


if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), use_reloader=False)
