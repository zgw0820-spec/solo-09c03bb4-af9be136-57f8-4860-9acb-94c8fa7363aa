# Gloo 梯度同步卡住演练台

数据分片调整后，多卡训练常出现“一侧已经跑完自己的分片、另一侧进入下一次梯度同步
却永远等不到对端”的挂死。本项目用 Flask 排障网页 + Docker Compose，在 **CPU +
PyTorch Gloo** 上用两个 rank 同步一个小模型的梯度，确定性复现该场景，并自动给出
卡点结论。

## 架构

```
浏览器 (轮询 /api/runs/<id>)
        │ HTTP
Flask 排障后端 (app/server.py，容器内由 tini/init 作 PID 1)
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
- 任何结束路径都先 `SIGTERM`（3s 宽限后 `SIGKILL`）回收两个训练进程；**两个 rank
  都退出后**再读一次事件文件，把“进程退出前已落盘的进度 + 回收结果”和结论一起
  原子写入 `runs/<run_id>/status.json`（临时文件 + `fsync` + `os.replace`）。
- **终态只发布一次、且不可变**：看门狗和停服收尾共用同一把锁，先到的一方写入，
  另一方直接得到同一份结果；网页第一次看到的终态、`status.json`、重启后用原链接
  回查三者逐字段一致。回收动作可能让对端 Gloo 报连接关闭，但这类 `sync_error`
  只进入进度快照，不会改写判定时已经定下的结论。
- **收尾期间服务仍可响应**：回收两个 rank 的这段时间里，详情页继续轮询，状态显示
  “收尾中（回收进程）”，两侧已落盘的进度照常可见；按钮暂时禁用。终态出现时一定
  已经带两条回收记录，不会先显示“已结束”再补回收记录。
- **服务停止（docker stop / SIGTERM）**：若看门狗已经判定、只差收尾，则沿用该结论
  发布；若演练仍在运行、尚未形成结论，则标记为 `interrupted`（服务中断）并附上
  两个 rank 的回收结果和中断前进度。服务被 `SIGKILL` 等来不及收尾时，回查会用
  事件文件还原一份“服务中断”结果，不让历史链接变成 404。

## 运行（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8080
```

演练数据持久化在命名卷 `drill-runs`（容器内 `/data/runs`）。`docker compose stop`
会给 Flask 发 SIGTERM（`init: true` + `stop_grace_period: 30s`），服务利用这段时间
回收两个 rank 并发布终态；再次 `docker compose start` 后，首页与详情页仍能查到
停止前的这轮演练及“服务中断”结论。

页面填写：两侧各自的样本批数（即同步步数）、可选“某一侧在第 N 步同步前暂停”、
演练超时秒数，然后启动。运行页每秒刷新，展示两侧**实际进入 / 完成的同步步**、
进程存活与最后事件；可随时“放行暂停的 rank”或“停止并回收”。结束后页面给出结论，
并可用相同参数一键重跑，也可回首页改参数重跑。

演示“暂停 → 停服 → 重启回查”：

```bash
docker compose up -d --build
# 浏览器发起：rank0=4 / rank1=4，暂停侧 rank0、暂停步 2，超时 120
docker compose stop            # 网页停在“收尾中”，随后服务退出
docker compose start           # 同一命名卷
# 用原链接打开 /runs/<run_id>：状态“服务中断”，带两个 rank 的回收结果与中断前进度
```

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
5. 运行中手动停止 → **第一次看到的终态就带两个 rank 回收记录**，收尾期间进度可见，
   重复回查同一份结果；
6. 非法参数 / 暂停步越界 → 400；
7. 分类器单元测试：现场阻塞、双完成、超时待查、双死异常、证据未成形继续等待；
8. 回收引发的 Gloo 报错（`peer_closed`）不改写已下的卡点结论；
9. SIGTERM 前演练仍在运行（为停服场景做准备）；
10. 看门狗与停服**同时**收尾 → 两个线程只发布一份终态，结论未被“服务中断”覆盖；
11. 缺少 `status.json`（甚至缺 `params.json`）的历史演练用事件还原，返回
    “服务中断”而不是 404，重复回查一致，非法 run id 不越界；
12. 停服：两个 rank 均回收后才发布终态，结论为 `interrupted`，`finalized_by=shutdown`；
13. 重启同一数据目录 → 原链接回查与停服时写入的终态逐字段一致，首页仍列出该演练；
14. 重启后回查已发布终态的演练（stuck）保持不变。

## 关键文件

| 文件 | 作用 |
| --- | --- |
| `app/trainer.py` | 单个 Gloo rank：小模型前向/反向 + all_reduce 梯度同步 + 暂停闸门 |
| `app/server.py` | Flask 路由、子进程管理、看门狗、判定、回收、终态发布、停服收尾、历史回查 |
| `app/templates/` | 发起页 / 实时进度与结论页（中文界面，1s 轮询，收尾期间继续展示进度） |
| `docker-compose.yml` / `Dockerfile` | 单容器两 rank（回环 Gloo），卷持久化演练数据，停止宽限期 30s |
| `tests/selftest.py` | 上述 14 个场景自测 |
