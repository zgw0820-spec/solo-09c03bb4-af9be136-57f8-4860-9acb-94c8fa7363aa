"""交付前自测：起真实 Flask 服务，跑关键场景并断言结论与进程回收。

运行（宿主机需有 torch + flask）:
    python -I tests/selftest.py
脚本通过 DRILL_* 环境变量把数据目录、解释器参数指向临时目录，不污染仓库。
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError("server did not open port in time")


def http(method: str, url: str, data: dict | None = None):
    body = None
    headers = {}
    if data is not None:
        body = urllib.parse.urlencode(data).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


import urllib.parse  # noqa: E402  (http() 依赖，置于调用前即可)


def wait_final(base_url: str, run_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        _, last = http("GET", f"{base_url}/api/runs/{run_id}")
        if last.get("status") != "running":
            return last
        time.sleep(0.3)
    raise AssertionError(f"run {run_id} 未在 {timeout}s 内结束，最后状态={last}")


def assert_reaped(final: dict, run_dir: Path) -> None:
    for rank in (0, 1):
        s = final["ranks"][str(rank)]
        assert s["alive"] is False, f"rank{rank} 未被回收"
    assert "reaped" in final and len(final["reaped"]) == 2, "缺少回收记录"
    # 直接向 pid 发 0 信号确认进程确实不存在
    for rank, info in final["reaped"].items():
        pid = info["pid"]
        try:
            os.kill(pid, 0)
            raise AssertionError(f"rank{rank} pid={pid} 仍然存在")
        except ProcessLookupError:
            pass
    for rank in (0, 1):
        p = run_dir / f"rank{rank}.events.jsonl"
        assert p.exists() and p.read_text().strip(), f"rank{rank} 事件文件为空"


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="gloo-drill-test-"))
    runs_dir = base / "runs"
    runs_dir.mkdir()
    port = 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    env = os.environ.copy()
    env.update(
        DRILL_BASE_DIR=str(runs_dir),
        DRILL_PYTHON=sys.executable,
        # 本地依赖装在 user site，-I 会屏蔽它；容器内 DRILL_PYTHON_FLAGS 默认带 -I
        DRILL_PYTHON_FLAGS="",
        PORT=str(port),
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "app.server"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}: {exc}")

    try:
        wait_port(port)
        base_url = f"http://127.0.0.1:{port}"

        # 场景 1：分片不均（2 vs 4），应判定 stuck，并指出卡在 rank1 第 3 步
        def t_stuck():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "2", "batches1": "4", "timeout_s": "20",
            })
            assert code == 200, body
            run_id = body["run_id"]
            final = wait_final(base_url, run_id, timeout=25)
            assert final["status"] == "stuck", final["status"]
            c = final["conclusion"]
            assert c["stuck_rank"] == 1 and c["finished_rank"] == 0, c
            assert c["blocked_step"] == 3 and c["finished_steps"] == 2, c
            r0, r1 = final["ranks"]["0"], final["ranks"]["1"]
            assert r0["training_complete"] and r0["entered"] == [1, 2] and r0["completed"] == [1, 2]
            assert 3 in r1["entered"] and 3 not in r1["completed"]
            assert_reaped(final, runs_dir / run_id)
        check("场景1 分片不均 -> 确认卡点(rank1 卡第3步, rank0 跑完2步)", t_stuck)

        # 场景 2：两侧批数相同，应正常跑完
        def t_completed():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "3", "batches1": "3", "timeout_s": "20",
            })
            assert code == 200, body
            final = wait_final(base_url, body["run_id"], timeout=25)
            assert final["status"] == "completed", final["status"]
            for rank in (0, 1):
                s = final["ranks"][str(rank)]
                assert s["entered"] == [1, 2, 3] and s["completed"] == [1, 2, 3], s
            assert_reaped(final, runs_dir / body["run_id"])
        check("场景2 两侧等批 -> 正常跑完且每步进/出一致", t_completed)

        # 场景 3：暂停 -> 观察到阻塞 -> 放行 -> 跑完
        def t_pause_resume():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "3", "batches1": "3",
                "pause_rank": "1", "pause_before_step": "2", "timeout_s": "20",
            })
            assert code == 200, body
            run_id = body["run_id"]
            # 等到 rank1 出现 paused 事件，且 rank0 已进入第 2 步同步（在等对端）
            deadline = time.time() + 15
            observed = {}
            while time.time() < deadline:
                _, st = http("GET", f"{base_url}/api/runs/{run_id}")
                if st["status"] != "running":
                    raise AssertionError(f"暂停期间演练提前结束: {st['status']}")
                r1 = st["ranks"]["1"]
                r0 = st["ranks"]["0"]
                if r1["paused_at"] == 2 and r0["last_event"] and r0["last_event"].get("event") == "sync_enter":
                    observed = st
                    break
                time.sleep(0.3)
            assert observed, "未观测到 rank1 暂停 / rank0 等待"
            assert observed["ranks"]["0"]["last_event"]["step"] == 2
            code, _ = http("POST", f"{base_url}/api/runs/{run_id}/resume", {})
            assert code == 200
            final = wait_final(base_url, run_id, timeout=25)
            assert final["status"] == "completed", final["status"]
            assert_reaped(final, runs_dir / run_id)
            # 事件流里应能看到 paused/resumed
            events = (runs_dir / run_id / "rank1.events.jsonl").read_text()
            assert '"event": "paused"' in events and '"event": "resumed"' in events
        check("场景3 一侧暂停后阻塞, 放行后恢复并跑完", t_pause_resume)

        # 场景 4：暂停且永不放行，超时 -> 原因待查
        def t_timeout():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "3", "batches1": "3",
                "pause_rank": "0", "pause_before_step": "2", "timeout_s": "6",
            })
            assert code == 200, body
            run_id = body["run_id"]
            final = wait_final(base_url, run_id, timeout=15)
            assert final["status"] == "timeout", final["status"]
            assert final["conclusion"].get("reason_pending") is True
            assert_reaped(final, runs_dir / run_id)
        check("场景4 暂停不放行超时 -> 证据不足/原因待查 + 回收", t_timeout)

        # 场景 5：运行中手动停止
        def t_manual_stop():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "100", "batches1": "100",
                "pause_rank": "0", "pause_before_step": "3", "timeout_s": "60",
            })
            assert code == 200, body
            run_id = body["run_id"]
            time.sleep(4)  # 让进程跑起来并进入暂停
            code, _ = http("POST", f"{base_url}/api/runs/{run_id}/stop", {})
            assert code == 200
            final = wait_final(base_url, run_id, timeout=15)
            assert final["status"] == "stopped", final["status"]
            assert_reaped(final, runs_dir / run_id)
        check("场景5 手动停止 -> 两个进程均回收", t_manual_stop)

        # 场景 6：参数校验与并发拒绝
        def t_validation():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "0", "batches1": "3", "timeout_s": "20",
            })
            assert code == 400, body
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "3", "batches1": "3",
                "pause_rank": "1", "pause_before_step": "9", "timeout_s": "20",
            })
            assert code == 400, body
        check("场景6 非法参数返回 400", t_validation)

        # 场景 7：分类器单元测试（构造事件摘要，覆盖现场阻塞/超时/异常/等待等分支）
        def t_classifier():
            import types
            from app.server import classify

            def summ(alive, entered, completed, last=None, complete=False, err=None, state=""):
                return {
                    "alive": alive, "entered": entered, "completed": completed,
                    "last_event": last, "training_complete": complete,
                    "error": err, "state": state,
                }
            enter3 = {"event": "sync_enter", "step": 3}
            # 7a 现场阻塞：rank0 跑完 2 步退出，rank1 活着卡在第 3 步
            h = types.SimpleNamespace(start_ts=time.time() - 5, timeout_s=30)
            verdict, c = classify(h, {
                "0": summ(False, [1, 2], [1, 2], complete=True, state="已完成全部批次"),
                "1": summ(True, [1, 2, 3], [1, 2], last=enter3, state="阻塞在第 3 步梯度同步"),
            })
            assert verdict == "stuck" and c["evidence_type"] == "observed_block", (verdict, c)
            assert c["stuck_rank"] == 1 and c["blocked_step"] == 3
            # 7b 两侧完成
            verdict, c = classify(h, {
                "0": summ(False, [1, 2], [1, 2], complete=True),
                "1": summ(False, [1, 2], [1, 2], complete=True),
            })
            assert verdict == "completed"
            # 7c 超时原因待查（都还活着、未完成、无卡点证据）
            h2 = types.SimpleNamespace(start_ts=time.time() - 60, timeout_s=30)
            verdict, c = classify(h2, {
                "0": summ(True, [1], [1]),
                "1": summ(True, [1], [1], last={"event": "paused", "step": 2}),
            })
            assert verdict == "timeout" and c["reason_pending"] is True
            # 7d 双死且证据不足 -> error
            verdict, c = classify(h, {
                "0": summ(False, [1], [1], state="已退出"),
                "1": summ(False, [1], [], last={"event": "sync_enter", "step": 2}, state="已退出"),
            })
            assert verdict == "error"
            # 7e 证据未成形 -> 继续等待（rank1 暂停、尚未进入第 3 步）
            verdict, c = classify(h, {
                "0": summ(False, [1, 2], [1, 2], complete=True),
                "1": summ(True, [1, 2], [1, 2], last={"event": "paused", "step": 3}),
            })
            assert verdict is None
        check("场景7 分类器分支(现场阻塞/完成/超时待查/异常/等待)", t_classifier)

        # 场景 8：服务收到 SIGTERM 时必须回收在跑的训练进程（模拟 docker stop）
        def t_sigterm_reap():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "20", "batches1": "20",
                "pause_rank": "0", "pause_before_step": "2", "timeout_s": "60",
            })
            assert code == 200, body
            run_id = body["run_id"]
            deadline = time.time() + 15
            while time.time() < deadline:
                _, st = http("GET", f"{base_url}/api/runs/{run_id}")
                if st["ranks"]["0"]["paused_at"] == 2:
                    break
                time.sleep(0.3)
            else:
                raise AssertionError("演练未进入暂停态")
            # 标记本 run 后给服务器发 SIGTERM（最后的 finally 会真正发送）
            t_sigterm_reap.run_dir = str(runs_dir / run_id)
        check("场景8 SIGTERM 前演练已进入暂停", t_sigterm_reap)

    finally:
        sigterm_run_dir = getattr(t_sigterm_reap, "run_dir", None)
        server.send_signal(signal.SIGTERM)
        try:
            out = server.communicate(timeout=15)[0]
        except subprocess.TimeoutExpired:
            server.kill()
            out = server.communicate()[0]
        # 确认 SIGTERM 后该 run 的 trainer 子进程全部消失
        if sigterm_run_dir:
            deadline = time.time() + 5
            leftover = ""
            while time.time() < deadline:
                leftover = subprocess.run(
                    ["pgrep", "-f", sigterm_run_dir], capture_output=True, text=True
                ).stdout.strip()
                if not leftover:
                    break
                time.sleep(0.3)
            assert not leftover, f"SIGTERM 后仍有训练进程残留: {leftover}"
        if failures:
            print("\n--- server output ---\n" + out[-4000:])

    print(f"\n数据目录: {runs_dir}")
    if failures:
        print(f"失败 {len(failures)} 项: {failures}")
        return 1
    print("全部自测通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
