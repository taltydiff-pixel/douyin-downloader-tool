FROM python:3.13-slim

# 安装 ffmpeg（PyAV 依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 Docker 缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载 Whisper tiny 模型
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN python -c "from faster_whisper import WhisperModel; model = WhisperModel('tiny', device='cpu', compute_type='int8', download_root='/app/models'); print('Whisper tiny model downloaded successfully')"

# 复制应用代码
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# 创建运行时目录
RUN mkdir -p downloads temp

# Railway 通过 PORT 环境变量指定端口
EXPOSE 8000

CMD ["sh", "-c", "python -c \"from waitress import serve; from app import app; import os; port = int(os.environ.get('PORT', 8000)); print(f'Starting on port {port}'); serve(app, host='0.0.0.0', port=port)\""]
