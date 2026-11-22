"""交付前自测：起真实 Flask 服务，跑关键场景并断言结论与进程回收。

运行（宿主机需有 torch + flask，建议用装了依赖的解释器直接运行）:
    python tests/selftest.py
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
import urllib.parse
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
        raw = exc.read().decode(errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw[:200]}


import urllib.parse  # noqa: E402  (http() 依赖，置于调用前即可)


def wait_final(base_url: str, run_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        _, last = http("GET", f"{base_url}/api/runs/{run_id}")
        if last.get("finalized") and last.get("status") != "running":
            return last
        time.sleep(0.3)
    raise AssertionError(f"run {run_id} 未在 {timeout}s 内结束，最后状态={last}")


def wait_until(fn, timeout: float, what: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"等待超时：{what}")


def assert_reaped(final: dict, run_dir: Path) -> None:
    for rank in (0, 1):
        s = final["ranks"][str(rank)]
        assert s["alive"] is False, f"rank{rank} 未被回收"
    assert final.get("reaped") and len(final["reaped"]) == 2, "缺少回收记录"
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


def start_server(runs_dir: Path, port: int, extra_env: dict | None = None):
    env = os.environ.copy()
    env.update(
        DRILL_BASE_DIR=str(runs_dir),
        DRILL_PYTHON=sys.executable,
        # 本地依赖装在 user site，-I 会屏蔽它；容器内 DRILL_PYTHON_FLAGS 默认带 -I
        DRILL_PYTHON_FLAGS="",
        PORT=str(port),
    )
    env.update(extra_env or {})
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.server"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    wait_port(port)
    return proc


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def read_status_file(runs_dir: Path, run_id: str) -> dict:
    return json.loads((runs_dir / run_id / "status.json").read_text(encoding="utf-8"))


def main() -> int:
    base = Path(tempfile.mkdtemp(prefix="gloo-drill-test-"))
    runs_dir = base / "runs"
    runs_dir.mkdir()
    port = free_port()

    server = start_server(runs_dir, port)
    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}: {exc}")

    try:
        base_url = f"http://127.0.0.1:{port}"

        # 场景 1：分片不均（2 vs 4），应判定 stuck，并指出卡在 rank1 第 3 步
        def t_stuck():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "2", "batches1": "4", "timeout_s": "20",
            })
            assert code == 200, body
            run_id = body["run_id"]
            t_stuck.run_id = run_id
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

        # 场景 5：运行中手动停止；终态第一次可见时就应带回收记录
        def t_manual_stop():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "100", "batches1": "100",
                "pause_rank": "0", "pause_before_step": "3", "timeout_s": "60",
            })
            assert code == 200, body
            run_id = body["run_id"]
            # 等到进程真正跑起来并进入暂停（进入过第 1、2 步同步），再停止
            wait_until(
                lambda: http("GET", f"{base_url}/api/runs/{run_id}")[1]["ranks"]["0"]["paused_at"] == 3,
                timeout=20, what="演练进入暂停态",
            )
            code, _ = http("POST", f"{base_url}/api/runs/{run_id}/stop", {})
            assert code == 200
            deadline = time.time() + 15
            final = None
            saw_finalizing = False
            while time.time() < deadline:
                _, st = http("GET", f"{base_url}/api/runs/{run_id}")
                if st.get("finalizing"):
                    saw_finalizing = True
                    assert st["status"] == "running", "收尾期间应仍可查看进度"
                    assert st["ranks"]["0"]["entered"], "收尾期间的进度应可见"
                if st.get("finalized"):
                    final = st
                    # 第一次看到终态就必须带回收记录，不能先给一份没有回收记录的结论
                    assert st.get("reaped") and len(st["reaped"]) == 2, st
                    break
                time.sleep(0.05)
            assert final is not None, "停止后未拿到终态"
            assert final["status"] == "stopped", final["status"]
            assert_reaped(final, runs_dir / run_id)
            # 同一份终态重复读取完全一致
            _, again = http("GET", f"{base_url}/api/runs/{run_id}")
            assert again == final, "重复回查终态不一致"
            # 网页首次看到的终态与落盘的 status.json 逐字段一致
            assert final == read_status_file(runs_dir, run_id), "内存终态与 status.json 不一致"
            t_manual_stop.saw_finalizing = saw_finalizing
        check("场景5 手动停止 -> 首次可见的终态即含两个 rank 回收记录", t_manual_stop)

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

        # 场景 8：回收引发的通信报错不得改写看门狗结论
        def t_reap_error_not_rewrite():
            import types
            from app.server import classify

            def summ(alive, entered, completed, last=None, complete=False, err=None, state=""):
                return {
                    "alive": alive, "entered": entered, "completed": completed,
                    "last_event": last, "training_complete": complete,
                    "error": err, "state": state,
                }
            h = types.SimpleNamespace(start_ts=time.time() - 5, timeout_s=30)
            # 判定发生时 rank1 仍活着；结论应为 stuck，而不是因为后续回收导致
            # 的 sync_error 被误判成 error（classify 只看判定时刻的快照）
            verdict, c = classify(h, {
                "0": summ(False, [1, 2], [1, 2], complete=True),
                "1": summ(True, [1, 2, 3], [1, 2], last={"event": "sync_enter", "step": 3}),
            })
            assert verdict == "stuck"
            # 判定后回收使 rank1 落盘 Gloo 报错：事件流出现 sync_error/exit，但
            # status.json 里已定下的结论保持 stuck（场景 1 的真实运行再加断言）
            verdict2, c2 = classify(h, {
                "0": summ(False, [1, 2], [1, 2], complete=True),
                "1": summ(False, [1, 2, 3], [1, 2],
                          last={"event": "exit", "code": 1}, complete=False,
                          err={"type": "RuntimeError", "message": "Connection reset by peer"},
                          state="同步报错（RuntimeError）"),
            })
            assert verdict2 == "stuck" and c2["evidence_type"] == "peer_closed", (verdict2, c2)
            final = read_status_file(runs_dir, t_stuck.run_id)
            assert final["status"] == "stuck", final["status"]
            assert final["conclusion"]["blocked_step"] == 3
        check("场景8 回收引发的 Gloo 报错不改写已下结论", t_reap_error_not_rewrite)

        # 场景 9：停服时保留看门狗结论；对仍在运行的演练标记服务中断
        sigterm_state: dict = {}

        def t_sigterm_publish():
            code, body = http("POST", f"{base_url}/api/runs", {
                "batches0": "20", "batches1": "20",
                "pause_rank": "0", "pause_before_step": "2", "timeout_s": "60",
            })
            assert code == 200, body
            run_id = body["run_id"]
            wait_until(
                lambda: http("GET", f"{base_url}/api/runs/{run_id}")[1]["ranks"]["0"]["paused_at"] == 2,
                timeout=15, what="演练进入暂停态",
            )
            status_path = runs_dir / run_id / "status.json"
            assert not status_path.exists(), "服务停止前不应有终态"
            sigterm_state["run_id"] = run_id

        # 场景 9 的发信号与重启回查放在 finally 之后继续做
        check("场景9 停服前演练仍在运行", t_sigterm_publish)

        # 场景 10：看门狗与停服同时收尾只发布一份终态（并发 finalize）
        def t_finalize_once():
            import threading as _threading
            from app.server import RunHandle, finalize

            run_dir = runs_dir / "concurrent-finalize"
            run_dir.mkdir()
            (run_dir / "params.json").write_text(
                json.dumps({"batches": [1, 1], "pause_rank": None,
                            "pause_before_step": None, "timeout_s": 30}),
                encoding="utf-8",
            )
            procs = {
                r: subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                for r in (0, 1)
            }
            handle = RunHandle(
                run_id="concurrent-finalize", run_dir=run_dir,
                params=json.loads((run_dir / "params.json").read_text()),
                start_ts=time.time(), timeout_s=30, procs=procs,
            )
            results: list[dict] = []
            barrier = _threading.Barrier(3)

            def run_finalize(verdict, conclusion):
                barrier.wait()
                results.append(finalize(handle, verdict, conclusion))

            # 看门狗在判定后、收尾前记下结论；停服路径据此沿用而不是改成“服务中断”
            handle.pending_verdict = ("stuck", {"title": "watchdog 结论", "blocked_step": 3})
            threads = [
                _threading.Thread(target=run_finalize, args=(
                    "stuck", {"title": "watchdog 结论", "blocked_step": 3},
                )),
                _threading.Thread(target=run_finalize, args=(None, None)),
            ]
            for th in threads:
                th.start()
            barrier.wait()
            for th in threads:
                th.join(timeout=20)
            status = read_status_file(runs_dir, "concurrent-finalize")
            assert len(results) == 2 and results[0] is results[1], "两次 finalize 返回了不同终态"
            assert status["status"] == "stuck", status["status"]
            assert status["conclusion"]["title"] == "watchdog 结论"
            assert status["finalized"] is True and len(status["reaped"]) == 2
            for proc in procs.values():
                assert proc.poll() is not None
        check("场景10 看门狗与停服同时收尾 -> 只发布一份终态", t_finalize_once)

        # 场景 11：无 status.json 的历史演练用事件还原，不返回“演练不存在”
        def t_recover_missing_status():
            run_dir = runs_dir / "20260101-000000-deadbeef"
            run_dir.mkdir()
            (run_dir / "params.json").write_text(json.dumps({
                "batches": [2, 4], "pause_rank": None,
                "pause_before_step": None, "timeout_s": 30,
            }), encoding="utf-8")
            (run_dir / "rank0.events.jsonl").write_text(
                "\n".join(json.dumps(e) for e in [
                    {"ts": 1000.0, "rank": 0, "event": "start"},
                    {"ts": 1001.0, "rank": 0, "event": "sync_enter", "step": 1},
                    {"ts": 1002.0, "rank": 0, "event": "sync_done", "step": 1},
                ]) + "\n", encoding="utf-8")
            (run_dir / "rank1.events.jsonl").write_text(
                "\n".join(json.dumps(e) for e in [
                    {"ts": 1000.0, "rank": 1, "event": "start"},
                    {"ts": 1001.0, "rank": 1, "event": "sync_enter", "step": 1},
                    {"ts": 1003.0, "rank": 1, "event": "sync_enter", "step": 2},
                ]) + "\n", encoding="utf-8")
            code, body = http("GET", f"{base_url}/api/runs/{run_dir.name}")
            assert code == 200, body
            assert body["status"] == "interrupted", body
            assert body["conclusion"].get("recovered") is True
            assert body["ranks"]["1"]["entered"] == [1, 2]
            # 重复回查一致，且不应凭空生成 status.json
            code2, body2 = http("GET", f"{base_url}/api/runs/{run_dir.name}")
            assert code2 == 200 and body2 == body
            assert not (run_dir / "status.json").exists()
            # 非法 run id 不会越界读取
            code3, _ = http("GET", f"{base_url}/api/runs/..%2F..%2Fetc")
            assert code3 == 404, code3
            # 只有事件、没有 params.json 的半残目录也能回查，而不是 404
            partial = runs_dir / "20260101-000000-cafebabe"
            partial.mkdir()
            (partial / "rank0.events.jsonl").write_text(
                json.dumps({"ts": 2000.0, "rank": 0, "event": "start"}) + "\n",
                encoding="utf-8",
            )
            code4, body4 = http("GET", f"{base_url}/api/runs/{partial.name}")
            assert code4 == 200 and body4["status"] == "interrupted", (code4, body4)
            assert body4["timeout_s"] is None
        check("场景11 缺少终态文件的历史演练可回查(服务中断)", t_recover_missing_status)

    finally:
        pass

    # ---- 停服：SIGTERM 时发布终态、回收两个 rank，重启后原链接回查一致 ----
    stop_failures = []
    try:
        run_id = sigterm_state["run_id"]
        server.send_signal(signal.SIGTERM)
        try:
            server.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.communicate()

        # 子进程全部回收
        leftover = subprocess.run(
            ["pgrep", "-f", str(runs_dir / run_id)], capture_output=True, text=True
        ).stdout.strip()
        assert not leftover, f"SIGTERM 后仍有训练进程残留: {leftover}"

        status = read_status_file(runs_dir, run_id)
        assert status["status"] == "interrupted", status["status"]
        assert status["conclusion"].get("interrupted") is True
        assert len(status["reaped"]) == 2, status.get("reaped")
        assert status["finalized_by"] == "shutdown", status["finalized_by"]
        # 两个 rank 都已回收后才发布：存活标记为 False，且保留了暂停前的进度
        for rank in (0, 1):
            s = status["ranks"][str(rank)]
            assert s["alive"] is False
            assert s["exit_code"] is not None
        assert status["ranks"]["0"]["paused_at"] == 2, status["ranks"]["0"]
        for rank, info in status["reaped"].items():
            try:
                os.kill(info["pid"], 0)
                raise AssertionError(f"rank{rank} pid={info['pid']} 仍然存在")
            except ProcessLookupError:
                pass
        print("PASS  场景12 停服：标记服务中断、两个 rank 回收后才发布终态")
    except Exception as exc:  # noqa: BLE001
        stop_failures.append("场景12 停服终态")
        print(f"FAIL  场景12 停服：标记服务中断、两个 rank 回收后才发布终态: {exc}")

    # ---- 重启同一个数据目录，用原链接回查 ----
    port2 = free_port()
    server2 = start_server(runs_dir, port2)
    base2 = f"http://127.0.0.1:{port2}"
    try:
        def t_restart_lookup():
            run_id = sigterm_state["run_id"]
            status = read_status_file(runs_dir, run_id)
            code, body = http("GET", f"{base2}/api/runs/{run_id}")
            assert code == 200, body
            # 重启后回查与停服时写入的终态逐字段一致
            assert body == status, "重启后回查与停服终态不一致"
            # 首页仍列出该演练
            with urllib.request.urlopen(f"{base2}/", timeout=10) as resp:
                html = resp.read().decode()
            assert run_id in html, "首页未列出停服时的演练"
            assert "服务中断" in html
            # 详情页链接可打开
            with urllib.request.urlopen(f"{base2}/runs/{run_id}", timeout=10) as resp:
                page = resp.read().decode()
            assert status["ranks"]["0"]["paused_at"] == 2
            assert "演练" in page
        check("场景13 重启后原链接回查得到同一份终态", t_restart_lookup)

        def t_restart_finalized_unchanged():
            run_id = t_stuck.run_id
            status = read_status_file(runs_dir, run_id)
            code, body = http("GET", f"{base2}/api/runs/{run_id}")
            assert code == 200, body
            assert body == status, "重启后终态与写入的 status.json 不一致"
            assert body["status"] == "stuck"
        check("场景14 重启后回查已发布终态的演练不变", t_restart_finalized_unchanged)
    finally:
        server2.send_signal(signal.SIGTERM)
        try:
            server2.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            server2.kill()
            server2.communicate()

    failures.extend(stop_failures)
    print(f"\n数据目录: {runs_dir}")
    if failures:
        print(f"失败 {len(failures)} 项: {failures}")
        return 1
    print("全部自测通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
