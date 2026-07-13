#!/bin/bash
set -e

echo "=== 抖音下载工具 v2.1 启动中 ==="

# 下载 Whisper tiny 模型（如果不存在）
# 下载 Whisper tiny 模型（使用 huggingface 镜像加速）
MODEL_CACHE="$HOME/.cache/huggingface/hub"
if [ ! -d "$MODEL_CACHE/models--Systran--faster-whisper-tiny" ]; then
    echo "[1/2] 首次启动，下载 Whisper tiny 模型（约 150MB）..."
    python -c "
import os, time
# 优先使用镜像站下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
max_retries = 3
for i in range(max_retries):
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel('tiny', device='cpu', compute_type='int8')
        print('Whisper tiny 模型下载完成')
        break
    except Exception as e:
        if i < max_retries - 1:
            print(f'下载失败 (尝试 {i+1}/{max_retries}): {e}, 5秒后重试...')
            time.sleep(5)
        else:
            print(f'警告: 模型下载失败，语音转文字功能不可用: {e}')
"
else
    echo "[1/2] Whisper 模型已缓存"
fi

echo "[2/2] 启动 Web 服务..."

# 使用 gunicorn 或 waitress
PORT="${PORT:-8000}"
if command -v gunicorn &> /dev/null; then
    exec gunicorn app:app -b "0.0.0.0:$PORT" -w 2 --timeout 300 -k gthread --threads 2
else
    python -c "from waitress import serve; from app import app; import os; port=int(os.environ.get('PORT',8000)); print(f'Server on port {port}'); serve(app, host='0.0.0.0', port=port)"
fi
