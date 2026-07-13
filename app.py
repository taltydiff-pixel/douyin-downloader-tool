#!/usr/bin/env python3
"""
抖音爆款视频下载工具 - 后端服务 v2.0
集成 TikHub API + DeepSeek AI + 语音转录 + 深度分析 + 爆款生成
"""

import os
import re
import json
import time
import hashlib
import subprocess
import requests
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from urllib.parse import quote

app = Flask(__name__)
CORS(app)

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

TIKHUB_API_URL = "https://api.tikhub.io/api/v1/douyin/web/fetch_one_video"

# ffmpeg 路径（云端使用系统PATH，本地使用同目录exe）
FFMPEG_PATH = "ffmpeg"  # Docker/Linux 使用系统PATH
if os.name == "nt":  # Windows 本地开发
    _local_ffmpeg = os.path.join(os.path.dirname(BASE_DIR), "ffmpeg.exe")
    if not os.path.exists(_local_ffmpeg):
        _local_ffmpeg = os.path.join(BASE_DIR, "..", "ffmpeg.exe")
    if os.path.exists(_local_ffmpeg):
        FFMPEG_PATH = _local_ffmpeg

# Whisper 模型路径（Docker 构建时预下载到 /app/models，本地使用缓存）
_cloud_model = os.path.join(BASE_DIR, "models")
if os.path.exists(_cloud_model):
    WHISPER_MODEL_PATH = _cloud_model
else:
    WHISPER_MODEL_PATH = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny/snapshots/abc123/"
    )

# DeepSeek API 配置（两个Key备用）
DEEPSEEK_CONFIGS = [
    {"key": "sk-7f488af31c6f456c8d2bbe3adf8f51e0", "name": "主Key"},
    {"key": "sk-a6a0c43c713c4033acdbc5df04e0e760", "name": "备用Key"},
]
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
_deepseek_key_index = 0


# ============ 辅助函数 ============

def extract_urls_from_text(text):
    """从混合文本中提取所有URL"""
    return re.findall(r'https?://[^\s<>"\'）\)\]】,\uff0c]+', text)


def resolve_short_url(short_url):
    """解析抖音短链接(v.douyin.com)，跟随重定向获取真实URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(short_url, headers=headers, allow_redirects=True, timeout=15)
        if resp.url and resp.url != short_url:
            return resp.url
        return None
    except Exception:
        return None


def extract_modal_id(text):
    """从文本或URL中提取modal_id（支持所有抖音链接格式）"""
    if not text:
        return None
    resolved_text = text
    for url in extract_urls_from_text(text):
        if 'v.douyin.com' in url:
            resolved = resolve_short_url(url)
            if resolved:
                resolved_text += "\n" + resolved
    m = re.search(r'modal_id[=:](\d+)', resolved_text)
    if m:
        return m.group(1)
    m = re.search(r'/video/(\d{16,})', resolved_text)
    if m:
        return m.group(1)
    m = re.search(r'(\d{16,})', resolved_text.strip())
    if m:
        return m.group(1)
    return None


def get_video_info_from_tikhub(modal_id, token=None):
    """通过 TikHub API 获取视频下载链接"""
    if not token:
        config_path = os.path.expanduser("~/.openclaw/config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            token = config.get("tikhub_api_token", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{TIKHUB_API_URL}?aweme_id={modal_id}&need_anchor_info=false"
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 200:
            return {"error": f"API 返回错误: {data.get('message', '未知错误')}", "modal_id": modal_id}
        aweme_detail = data.get("data", {}).get("aweme_detail")
        if not aweme_detail:
            return {"error": "API 响应中缺少视频信息", "modal_id": modal_id}
        video = aweme_detail.get("video", {})
        download_addr = video.get("download_addr", {})
        url_list = download_addr.get("url_list", [])
        if url_list:
            return {"modal_id": modal_id, "video_url": url_list[0]}
        play_addr = video.get("play_addr", {})
        url_list = play_addr.get("url_list", [])
        if url_list:
            return {"modal_id": modal_id, "video_url": url_list[0]}
        return {"error": "无法从响应中提取视频链接", "modal_id": modal_id}
    except requests.exceptions.RequestException as e:
        return {"error": f"请求失败: {str(e)}", "modal_id": modal_id}


def download_video_file(video_url, output_path):
    """下载视频文件到本地"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.douyin.com/",
    }
    try:
        resp = requests.get(video_url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        file_size = os.path.getsize(output_path)
        return {"success": True, "path": output_path, "size": file_size}
    except requests.exceptions.RequestException as e:
        return {"error": f"下载失败: {str(e)}"}


def call_deepseek(messages, temperature=0.7, max_tokens=2048, timeout=120):
    """调用 DeepSeek API（自动轮换Key）"""
    global _deepseek_key_index
    for attempt in range(len(DEEPSEEK_CONFIGS)):
        idx = (_deepseek_key_index + attempt) % len(DEEPSEEK_CONFIGS)
        cfg = DEEPSEEK_CONFIGS[idx]
        headers = {
            "Authorization": f"Bearer {cfg['key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        try:
            resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            _deepseek_key_index = idx
            return {"success": True, "content": data["choices"][0]["message"]["content"]}
        except Exception as e:
            if attempt == len(DEEPSEEK_CONFIGS) - 1:
                return {"error": f"AI 调用失败: {str(e)}"}
            continue
    return {"error": "所有 API Key 均不可用"}


def get_safe_filename(modal_id, fmt="mp4"):
    """生成安全的文件名"""
    timestamp = int(time.time())
    return f"douyin_{modal_id}_{timestamp}.{fmt}"


def extract_audio_from_video(video_path, audio_path):
    """用PyAV从视频中提取音频（无需外部ffmpeg）"""
    if not os.path.exists(video_path):
        return {"error": f"视频文件不存在: {video_path}"}
    try:
        import av
        container = av.open(video_path)
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
        if audio_stream is None:
            return {"error": "视频中没有音频流"}
        output = av.open(audio_path, 'w', format='wav')
        out_stream = output.add_stream('pcm_s16le', rate=16000, layout='mono')
        for frame in container.decode(audio=0):
            for packet in out_stream.encode(frame):
                output.mux(packet)
        for packet in out_stream.encode(None):
            output.mux(packet)
        output.close()
        container.close()
        if os.path.exists(audio_path):
            return {"success": True, "path": audio_path}
        return {"error": "音频文件未生成"}
    except Exception as e:
        return {"error": f"音频提取异常: {str(e)}"}


def transcribe_audio(audio_path):
    """使用 faster-whisper 进行语音识别"""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(
            WHISPER_MODEL_PATH, device='cpu', compute_type='int8'
        )
        segments, info = model.transcribe(
            audio_path, language='zh', beam_size=3
        )
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text)
        return {"success": True, "text": ''.join(text_parts)}
    except Exception as e:
        return {"error": f"语音识别失败: {str(e)}"}


# ============ API 路由 ============

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/parse", methods=["POST"])
def parse_video():
    """解析视频链接，返回视频信息"""
    data = request.get_json()
    url = data.get("url", "").strip()
    token = data.get("token", "")
    if not url:
        return jsonify({"success": False, "error": "请输入抖音视频链接"}), 400
    modal_id = extract_modal_id(url)
    if not modal_id:
        return jsonify({"success": False, "error": "无法识别视频链接，请确认是有效的抖音分享链接"}), 400
    info = get_video_info_from_tikhub(modal_id, token if token else None)
    if "error" in info:
        return jsonify({"success": False, "error": info["error"]}), 400
    # 简短AI分析
    ai_analysis = None
    try:
        ai_result = call_deepseek([
            {"role": "system", "content": "你是一个抖音视频分析助手。"},
            {"role": "user", "content": f"抖音视频ID: {modal_id}。请生成一段简短的视频描述（50字以内）和3-5个推荐标签。"}
        ], max_tokens=500)
        if ai_result.get("success"):
            ai_analysis = ai_result["content"]
    except Exception:
        pass
    return jsonify({
        "success": True, "modal_id": modal_id,
        "video_url": info["video_url"],
        "ai_analysis": ai_analysis
    })


@app.route("/api/download", methods=["POST"])
def download_video_route():
    """下载视频并返回文件"""
    data = request.get_json()
    video_url = data.get("video_url", "").strip()
    modal_id = data.get("modal_id", "")
    file_format = data.get("format", "mp4")
    if not video_url:
        return jsonify({"success": False, "error": "缺少视频链接"}), 400
    fmt_map = {
        "mp4": {"ext": "mp4", "mime": "video/mp4"},
        "webm": {"ext": "webm", "mime": "video/webm"},
        "avi": {"ext": "avi", "mime": "video/x-msvideo"},
    }
    fmt_info = fmt_map.get(file_format, fmt_map["mp4"])
    filename = get_safe_filename(modal_id, fmt_info["ext"])
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    result = download_video_file(video_url, output_path)
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 500
    return jsonify({
        "success": True, "filename": filename, "size": result["size"],
        "download_url": f"/api/file/{filename}", "format": file_format
    })


@app.route("/api/file/<filename>")
def serve_file(filename):
    """提供下载文件"""
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": "文件不存在或已过期"}), 404
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp4"
    mime_map = {"mp4": "video/mp4", "webm": "video/webm", "avi": "video/x-msvideo"}
    return send_file(
        file_path, mimetype=mime_map.get(ext, "application/octet-stream"),
        as_attachment=True, download_name=filename
    )


@app.route("/api/ai/analyze", methods=["POST"])
def ai_analyze():
    """使用 DeepSeek AI 分析视频内容（简单版）"""
    data = request.get_json()
    modal_id = data.get("modal_id", "")
    url = data.get("url", "")
    user_prompt = data.get("prompt", "")
    content = f"抖音视频ID: {modal_id}\n视频链接: {url}\n"
    if user_prompt:
        content += f"\n用户需求: {user_prompt}"
    messages = [
        {"role": "system", "content": "你是一个专业的抖音爆款视频分析助手。请用简洁专业的风格回答。"},
        {"role": "user", "content": content}
    ]
    result = call_deepseek(messages)
    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 500
    return jsonify({"success": True, "content": result["content"]})


@app.route("/api/config/token", methods=["POST"])
def save_token():
    """保存用户自定义的 TikHub Token"""
    data = request.get_json()
    token = data.get("token", "").strip()
    if not token:
        return jsonify({"success": False, "error": "Token 不能为空"}), 400
    config_dir = os.path.expanduser("~/.openclaw")
    config_path = os.path.join(config_dir, "config.json")
    os.makedirs(config_dir, exist_ok=True)
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    config["tikhub_api_token"] = token
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return jsonify({"success": True, "message": "Token 已保存"})


# ============ 新增 API：获取口播文案 ============

@app.route("/api/transcribe", methods=["POST"])
def transcribe_video():
    """
    获取口播文案：解析链接 → 下载视频 → 提取音频 → 语音识别 → AI清洗
    支持传入原始分享文本（自动解析）或 video_url
    """
    data = request.get_json()
    modal_id = data.get("modal_id", "")
    video_url = data.get("video_url", "")
    raw_url = data.get("url", "")  # 支持传入原始分享文本

    # 如果有原始文本，先解析获取新鲜URL
    if raw_url:
        parsed = extract_modal_id(raw_url)
        if parsed:
            modal_id = parsed
        token = None
        config_path = os.path.expanduser("~/.openclaw/config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            token = config.get("tikhub_api_token", "")
        info = get_video_info_from_tikhub(modal_id, token)
        if "error" not in info:
            video_url = info["video_url"]

    if not modal_id or not video_url:
        return jsonify({"success": False, "error": "无法获取视频信息，请先解析"}), 400

    try:
        # 1. 下载视频
        video_filename = f"temp_{modal_id}_{int(time.time())}.mp4"
        video_path = os.path.join(TEMP_DIR, video_filename)
        d_result = download_video_file(video_url, video_path)
        if "error" in d_result:
            return jsonify({"success": False, "error": f"视频下载失败: {d_result['error']}"}), 500

        # 2. 提取音频
        ts = int(time.time())
        audio_filename = f"temp_{modal_id}_{ts}.wav"
        audio_path = os.path.join(TEMP_DIR, audio_filename)
        a_result = extract_audio_from_video(video_path, audio_path)
        if "error" in a_result:
            return jsonify({"success": False, "error": a_result["error"]}), 500

        # 3. 语音识别
        t_result = transcribe_audio(audio_path)
        if "error" in t_result:
            return jsonify({"success": False, "error": t_result["error"]}), 500

        raw_text = t_result["text"]

        # 4. AI 清洗：去语气词，梳理通顺
        clean_result = call_deepseek([
            {
                "role": "system",
                "content": "你是一个文案整理助手。将以下语音识别文本进行清洗："
                           "1) 去掉所有语气词（啊、呢、吧、哦、嗯、那个、这个、就是说等）"
                           "2) 修正识别错误的断句和标点"
                           "3) 保留所有核心信息和论点，不做摘要压缩"
                           "4) 用流畅的书面化表达输出"
                           "5) 分段清晰，每段表达一个完整意思"
            },
            {"role": "user", "content": raw_text}
        ], temperature=0.3, max_tokens=2048)

        cleaned_text = raw_text
        if clean_result.get("success"):
            cleaned_text = clean_result["content"]

        # 清理临时文件
        try:
            os.remove(video_path)
            os.remove(audio_path)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "modal_id": modal_id,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "char_count": len(cleaned_text)
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"处理失败: {str(e)}"}), 500


# ============ 新增 API：深度AI分析 ============

@app.route("/api/analyze-deep", methods=["POST"])
def analyze_deep():
    """
    深度分析口播文案：
    - 目标用户画像
    - 核心话题
    - 爆款原因分析
    - 可引用的经典句子（客观无导向）
    - 保险从业者借鉴建议
    """
    data = request.get_json()
    transcript = data.get("transcript", "")

    if not transcript:
        return jsonify({"success": False, "error": "缺少口播文案内容"}), 400

    prompt = f"""你是一位专业的短视频内容分析专家。请对以下口播文案进行深度分析，输出结构化分析报告：

## 待分析的文案
{transcript}

## 分析要求
请按照以下结构输出：

### 🎯 目标用户画像
分析这段视频的目标受众是谁（年龄、职业、收入水平、认知水平等）

### 💬 核心话题
提炼视频讨论的核心话题和核心观点

### 🔥 爆款原因分析
分析为什么这个视频可能成为爆款（从情绪触发、认知差、信息密度、表达方式等角度）

### 📝 可直接引用的经典句子
摘取文案中客观、无导向型的经典表述句子（3-5句），方便直接引用

### 💡 保险从业者创作建议
如果保险从业者想参照这个视频生成自己的新视频文案，可以：
1. 从哪些角度切入
2. 借鉴什么结构
3. 如何结合自身业务做改编
4. 注意事项

请用专业、简洁的风格输出。"""

    result = call_deepseek([
        {"role": "system", "content": "你是一位资深的短视频内容策略分析师，擅长从内容中提取可复用的方法论。"},
        {"role": "user", "content": prompt}
    ], temperature=0.5, max_tokens=3072, timeout=150)

    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 500

    return jsonify({"success": True, "content": result["content"]})


# ============ 新增 API：一键生成爆款 ============

@app.route("/api/generate-viral", methods=["POST"])
def generate_viral():
    """
    一键生成爆款：基于原视频内容，生成全套新内容
    - 新口播稿 (参考 short-video-script)
    - 爆款标题 (参考 viral-title-generator)
    - 爆款口播文案 (参考 yijian-koubo-wenan)
    - 3秒钩子 (参考 shortvideo-hook)
    """
    data = request.get_json()
    transcript = data.get("transcript", "")
    topic = data.get("topic", "保险与财富管理")

    if not transcript:
        return jsonify({"success": False, "error": "缺少口播文案内容"}), 400

    system_prompt = """你是一位顶级的短视频爆款内容创作专家，精通多平台流量算法和爆款内容底层逻辑。
你擅长将已有的优质内容进行二次创作，生成全新的爆款素材。"""

    user_prompt = """请基于以下原始口播文案，生成一套完整的爆款内容套餐。

## 原始文案
{transcript}

## 生成要求

### 一、【新口播稿】- 参考短视频脚本创作方法论
创作一篇全新的口播稿（约500-650字），要求：
- 黄金3秒开头（悬念式/提问式/冲突式）
- 正文3个核心论点，逻辑递进
- 结尾总结+引导互动
- 口语化，适合直接录制

### 二、【爆款标题】- 参考自媒体爆款标题公式
生成5个抖音爆款标题（每个15-20字）：
- 1个数字型："{{数字}}个{{方法}}，{{时间}}内{{结果}}"
- 1个疑问型：制造好奇
- 1个反差型：前后对比
- 1个悬念型：引发点击
- 1个指令型：包含"一定要/千万别"等指令词

### 三、【爆款口播文案】- 参考爆款文案生成方法论
创作一篇结构完整的口播文案（550-680字），包含：
- 黄金开场钩子
- 情绪共鸣段落
- 认知反转/核心观点
- 3条实操落地方法
- 故事化表达
- 评论引导+关注话术

### 四、【3秒钩子】- 参考短视频钩子生成
生成5个黄金3秒开场钩子：
- 1个悬念型钩子
- 1个痛点型钩子
- 1个利益型钩子
- 1个反差型钩子
- 1个故事型钩子

请按以上四个板块清晰输出，每个板块用 ### 分隔。""".format(transcript=transcript)

    result = call_deepseek([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], temperature=0.7, max_tokens=4096, timeout=180)

    if "error" in result:
        return jsonify({"success": False, "error": result["error"]}), 500

    return jsonify({"success": True, "content": result["content"]})


# ============ 启动 ============
if __name__ == "__main__":
    print("=" * 60)
    print("  抖音爆款视频下载工具 v2.0")
    print("=" * 60)
    print(f"  DeepSeek AI: {'已配置' if DEEPSEEK_CONFIGS[0]['key'] else '未配置'}")
    print(f"  Whisper 模型: {'就绪' if os.path.exists(WHISPER_MODEL_PATH) else '未找到'}")
    print(f"  ffmpeg: {'就绪' if os.path.exists(FFMPEG_PATH) else '未找到'}")
    print()
    print("  新功能:")
    print("  1. 获取口播文案 - 视频→音频→文字→AI清洗")
    print("  2. 深度AI分析 - 目标用户/爆款原因/经典句子")
    print("  3. 一键生成爆款 - 新口播稿/标题/文案/钩子")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
