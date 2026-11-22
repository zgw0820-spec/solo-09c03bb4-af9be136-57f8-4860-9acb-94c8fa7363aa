FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    DRILL_BASE_DIR=/data/runs

WORKDIR /app

# 先装依赖，利用镜像层缓存
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app /app/app
COPY tests /app/tests

RUN mkdir -p /data/runs

EXPOSE 8080

# Flask 以 PID 1 运行，训练 rank 是它的子进程，停止容器时可统一回收
CMD ["python", "-m", "app.server"]
