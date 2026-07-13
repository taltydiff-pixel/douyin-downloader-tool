FROM python:3.13-slim

# ===== 安装系统依赖 =====
# ffmpeg: 运行时音频处理
# ffmpeg-dev 系列: PyAV 编译时需要
# libsndfile1: faster-whisper 依赖
# curl: 健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    libsndfile1 \
    libsndfile1-dev \
    pkg-config \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ===== 安装 Python 依赖 =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== 复制应用代码 =====
COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY entrypoint.sh .

# ===== 初始化 =====
RUN mkdir -p downloads temp models && chmod +x entrypoint.sh

EXPOSE 8000

# 入口脚本：首次启动时下载 Whisper 模型，然后启动服务
ENTRYPOINT ["./entrypoint.sh"]
