# Gloo 梯度同步卡住演练台

数据分片调整后，多卡训练常出现“一侧已经跑完自己的分片、另一侧进入下一次梯度同步
却永远等不到对端”的挂死。本项目用 Flask 排障网页 + Docker Compose，在 **CPU +
PyTorch Gloo** 上用两个 rank 同步一个小模型的梯度，确定性复现该场景，并自动给出
卡点结论。

## 架构

```
浏览器 (轮询 /api/runs/<id>)
        │ HTTP
Flask 排障后端 (app/server.py, PID 1)
   ├── 看门狗线程：读 JSONL 事件文件 → 还原进度 → 判定 → 回收进程 → 写结论
   ├── 子进程 rank0 (app/trainer.py, Gloo)
   └── 子进程 rank1 (app/trainer.py, Gloo)
        └── 两个 rank 经容器回环 127.0.0.1 做 all_reduce + barrier
```

- 每个 rank 把 `start / init_done / sync_enter / sync_done / paused / resumed /
  training_complete / sync_error / exit` 等事件以 JSON Lines 实时 fsync 落盘到
  `runs/<run_id>/rankN.events.jsonl`。后端只依赖事件文件，不依赖退出码。
- 判定逻辑（`classify`）：
  - 一侧 `training_complete`，另一侧**已进入**对端没到过的下一步且该步没有
    `sync_done` → `stuck`，报告卡住 rank、卡住步、跑完的 rank；
    - 对端仍存活且最后事件是 `sync_enter`：证据类型 `observed_block`（当场抓到阻塞）；
    - 对端已被 Gloo 报连接关闭而退出：证据类型 `peer_closed`（事后由事件流还原）；
  - 两侧都完成 → `completed`；
  - 达到超时仍不满足上述证据 → `timeout`，结论标注 **“原因待查”**；
  - 两进程都死且无有效证据 → `error`；手动停止 → `stopped`。
- 任何结束路径都先 `SIGTERM`（5s 宽限后 `SIGKILL`）回收两个训练进程，回收后再读
  一次事件，把结论写入 `runs/<run_id>/status.json`。服务收到 SIGTERM/SIGINT 时
  同样先回收子进程再退出（docker stop 不会留孤儿）。

## 运行（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8080
```

页面填写：两侧各自的样本批数（即同步步数）、可选“某一侧在第 N 步同步前暂停”、
演练超时秒数，然后启动。运行页每秒刷新，展示两侧**实际进入 / 完成的同步步**、
进程存活与最后事件；可随时“放行暂停的 rank”或“停止并回收”。结束后页面给出结论，
并可用相同参数一键重跑，也可回首页改参数重跑。

演练数据持久化在命名卷 `drill-runs`（容器内 `/data/runs`）。

## 本地直接运行（不用 Docker）

```bash
pip install -r requirements.txt
DRILL_PYTHON_FLAGS="" python -m app.server     # 默认 :8080，数据在 ./runs
python tests/selftest.py                        # 关键场景自测
```

## 自测覆盖（tests/selftest.py，真实起 Flask + 两个 Gloo rank）

1. 分片不均 2 vs 4 → 判定 stuck：rank1 卡第 3 步，rank0 跑完 2 步，进程全回收；
2. 两侧等批 3 vs 3 → completed，每步进/出一致；
3. rank1 第 2 步前暂停 → 现场看到 rank0 阻塞在同步 → 放行 → 跑完；
4. 暂停不放行 → 超时 → 结论“原因待查”并回收；
5. 运行中手动停止 → 两个进程均回收；
6. 非法参数 / 暂停步越界 → 400；
7. 分类器单元测试：现场阻塞、双完成、超时待查、双死异常、证据未成形继续等待；
8. 服务 SIGTERM（模拟 docker stop）→ 在跑的训练进程无残留。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `app/trainer.py` | 单个 Gloo rank：小模型前向/反向 + all_reduce 梯度同步 + 暂停闸门 |
| `app/server.py` | Flask 路由、子进程管理、看门狗、判定、回收、重跑 |
| `app/templates/` | 发起页 / 实时进度与结论页（中文界面，1s 轮询） |
| `docker-compose.yml` / `Dockerfile` | 单容器两 rank（回环 Gloo），卷持久化演练数据 |
| `tests/selftest.py` | 上述 8 个关键场景自测 |
