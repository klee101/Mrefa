from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
import numpy as np
import tempfile
import os
import uuid
import json
import requests
import logging
from io import BytesIO
import re
from typing import Dict, List, Optional
from pydantic import BaseModel
import aiofiles
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic 模型
class EmotionAnalysis(BaseModel):
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]


class AnalysisResponse(BaseModel):
    success: bool
    input_text: str
    response_text: str
    emotion_analysis: EmotionAnalysis
    expression_parameters: List[float]
    pose_parameters: List[float]
    audio_generated: bool
    audio_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str


class APIInfo(BaseModel):
    message: str
    endpoints: Dict[str, str]
    usage: Dict[str, Dict]


# 创建FastAPI应用
app = FastAPI(
    title="Voice Emotion Analysis API",
    description="AI-powered voice emotion analysis with FLAME parameter generation",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 线程池执行器用于异步执行同步操作
executor = ThreadPoolExecutor(max_workers=4)


class VoiceEmotionAPI:
    def __init__(self):
        self.api_config = {
            'type': 'openai',
            'base_url': 'https://api.siliconflow.cn/v1',
            'api_key': 'sk-xwqerljnfjdihzjtoirvovjutilixrjgdycmlqvdftpmbnnt',
            'model': 'deepseek-ai/DeepSeek-V3'
        }

        self.system_prompt = """你是一个友好、有情感的AI助手。请遵循以下规则：
        1. 用自然、温暖的语调回复
        2. 回复要简洁，控制在30字以内
        3. 根据用户的情感状态调整回复风格
        4. 支持中文和英文对话
        5. 输出格式要求：仅返回纯文本对话内容，排除所有非对话元素（如元数据、XML标签、markdown格式、表情符号、括号备注、系统提示信息等）"""

        self.emotion_prompt = """请严格分析以下文本的情感，只返回一个JSON对象，不要有任何其他文字。JSON格式如下：
        {
            "primary_emotion": "joy",
            "confidence": 0.85,
            "all_emotions": {
                "joy": 0.85,
                "sadness": 0.05,
                "anger": 0.02,
                "fear": 0.01,
                "surprise": 0.03,
                "disgust": 0.01,
                "neutral": 0.03
            }
        }
        文本："""

        self.setup_emotion_mapping()
        self.setup_recognizer()

    def setup_recognizer(self):
        """设置语音识别"""
        self.recognizer = sr.Recognizer()

    def setup_emotion_mapping(self):
        """设置情感到FLAME表情参数和姿态参数的映射"""
        # 更新的表情参数
        self.emotion_to_expression = {
            'joy': np.array([
                2.60, -0.35, 0.07, 0.52, 0.62, 0.26, 0.14, -0.31, -0.52, 0.86,
                -0.77, 0.26, 0.27, 0.41, -0.63, -0.31, -0.28, -0.28, 0.07, 0.39,
                0.27, -0.30, 0.33, -0.04, -0.20, 0.16, 0.03, 0.05, -0.42, 0.11,
                0.01, 0.05, 0.01, -0.09, 0.11, -0.02, 0.04, -0.11, 0.04, 0.08,
                -0.32, 0.04, 0.14, -0.17, -0.02, 0.10, -0.02, 0.06, 0.14, 0.03
            ]),
            'sadness': np.array([
                -0.22, -0.24, 0.20, 0.13, 0.59, 0.50, -0.93, 0.19, -0.38, 0.67,
                -0.65, 0.45, -0.51, -0.16, -0.29, -0.34, 0.01, -0.36, 0.10, 0.27,
                0.19, -0.22, 0.09, 0.26, 0.36, -0.04, 0.20, -0.20, -0.06, 0.01,
                0.03, 0.27, 0.18, -0.10, 0.21, -0.17, 0.13, -0.18, 0.17, 0.35,
                0.01, -0.12, 0.15, -0.02, 0.09, 0.16, -0.25, 0.18, -0.09, 0.21
            ]),
            'anger': np.array([
                0.68, 0.47, 0.31, 0.71, -0.36, -0.08, -0.25, 0.64, 0.21, -0.75,
                0.96, 0.44, -0.20, -0.26, 0.12, 0.15, 0.39, -0.26, 0.38, -0.07,
                0.07, -0.43, 0.12, -0.25, -0.10, 0.04, 0.02, 0.39, 0.32, 0.29,
                -0.07, -0.17, -0.22, 0.38, 0.17, 0.14, -0.01, -0.18, -0.06, -0.52,
                -0.41, -0.02, 0.08, -0.02, -0.08, -0.22, 0.12, 0.09, -0.07, -0.03
            ]),
            'fear': np.array([
                -0.21, -0.02, 0.10, 0.05, 0.55, 0.03, -0.33, 0.54, -0.47, 0.25,
                -0.03, 0.02, 0.06, 0.27, -0.15, 0.29, 0.11, 0.04, 0.06, -0.19,
                0.10, 0.38, -0.39, 0.00, 0.28, -0.17, 0.06, 0.02, -0.14, 0.01,
                -0.07, -0.05, 0.10, -0.05, -0.17, -0.13, -0.08, -0.05, 0.23, 0.06,
                -0.29, -0.13, -0.05, -0.03, -0.18, -0.08, -0.24, 0.20, -0.13, 0.00
            ]),
            'surprise': np.array([
                -0.50, 0.10, 0.15, 0.35, 0.86, 0.32, -0.51, 0.71, -0.59, 0.38,
                0.04, 0.50, 0.10, 0.30, -0.48, 0.21, 0.29, -0.12, 0.27, -0.19,
                0.09, 0.09, -0.08, -0.03, 0.11, -0.08, 0.16, 0.03, -0.08, 0.13,
                0.02, 0.04, -0.12, 0.08, -0.16, -0.04, -0.17, -0.10, 0.16, 0.01,
                -0.35, -0.03, 0.07, 0.02, -0.10, -0.17, -0.21, 0.20, -0.07, 0.11
            ]),
            'disgust': np.array([
                0.19, 0.13, 0.31, 1.06, -0.92, -0.20, 0.41, -0.23, 0.76, -0.02,
                -0.37, 0.63, -0.47, 0.06, 0.31, -0.77, -0.43, -0.41, -0.17, 0.63,
                0.38, -0.72, 0.86, 0.00, -0.31, 0.58, -0.04, 0.46, -0.03, 0.35,
                -0.09, -0.09, -0.06, -0.14, 0.52, -0.05, 0.18, -0.21, -0.16, 0.01,
                -0.14, 0.07, 0.42, -0.13, 0.10, 0.04, 0.00, 0.09, 0.11, 0.17
            ]),
            'neutral': np.zeros(50)
        }

        # 新增姿态参数映射
        self.emotion_to_pose = {
            'joy': np.array([0.12, 0.01, 0.00, 0.25, 0.00, -0.01]),
            'sadness': np.array([0.12, -0.03, 0.00, 0.02, 0.00, 0.02]),
            'anger': np.array([0.29, 0.02, 0.02, -0.03, 0.00, -0.02]),
            'fear': np.array([0.02, 0.00, 0.03, 0.41, 0.01, -0.01]),
            'surprise': np.array([0.05, -0.02, 0.03, 0.37, 0.01, -0.02]),
            'disgust': np.array([0.18, -0.06, -0.01, 0.00, 0.00, 0.03]),
            'neutral': np.zeros(6)
        }

        self.emotion_to_voice_prompt = {
            'joy': "Can you say it with a happy and cheerful emotion?",
            'sadness': "Can you say it with a sad and gentle emotion?",
            'anger': "Can you say it with an angry and stern emotion?",
            'fear': "Can you say it with a worried and nervous emotion?",
            'surprise': "Can you say it with a surprised and amazed emotion?",
            'disgust': "Can you say it with a disgusted and annoyed emotion?",
            'neutral': "Can you say it with a calm and neutral emotion?"
        }

    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return 'chinese' if len(chinese_chars) > 0 else 'english'

    async def speech_to_text(self, audio_file_path: str) -> str:
        """使用SiliconFlow API进行语音转文本"""
        try:
            # 读取音频文件
            async with aiofiles.open(audio_file_path, 'rb') as f:
                audio_data = await f.read()

            # 准备API请求
            url = "https://api.siliconflow.cn/v1/audio/transcriptions"

            # 构建multipart/form-data
            files = {
                'file': ('audio.wav', audio_data, 'audio/wav'),
                'model': (None, 'FunAudioLLM/SenseVoiceSmall'),
                'language': (None, 'auto'),  # 自动检测语言
                'response_format': (None, 'json')
            }

            headers = {
                "Authorization": f"Bearer {self.api_config['api_key']}"
            }

            # 使用线程池异步发送请求
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: requests.post(url, files=files, headers=headers, timeout=30)
            )

            if response.status_code == 200:
                result = response.json()
                # SenseVoice API返回的文本在'text'字段中
                text = result.get('text', '').strip()
                if text:
                    return text
                else:
                    return "无法识别语音内容"
            else:
                logger.error(f"语音识别API调用失败: {response.status_code}, {response.text}")
                return "语音识别失败"

        except Exception as e:
            logger.error(f"语音识别错误: {e}")
            return "语音识别失败"

    async def call_llm_api(self, user_input: str) -> str:
        """异步调用大模型API生成回复"""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]

            headers = {
                'Authorization': f'Bearer {self.api_config["api_key"]}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': self.api_config['model'],
                'messages': messages,
                'max_tokens': 150,
                'temperature': 0.7
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: requests.post(
                    f"{self.api_config['base_url']}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return self._fallback_response(user_input)

        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return self._fallback_response(user_input)

    def _fallback_response(self, input_text: str) -> str:
        """备用回复"""
        input_lower = input_text.lower()
        if any(word in input_lower for word in ['你好', 'hello', 'hi']):
            return "你好！很高兴和你聊天！"
        elif any(word in input_lower for word in ['开心', 'happy']):
            return "我也感到很开心！"
        else:
            return "我明白了，请继续分享你的想法。"

    async def analyze_emotion(self, text: str) -> dict:
        """异步分析文本情感"""
        try:
            emotion_request = self.emotion_prompt + text

            messages = [
                {"role": "system", "content": "你是一个专业的情感分析助手，只返回JSON格式的分析结果。"},
                {"role": "user", "content": emotion_request}
            ]

            headers = {
                'Authorization': f'Bearer {self.api_config["api_key"]}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': self.api_config['model'],
                'messages': messages,
                'max_tokens': 200,
                'temperature': 0.1
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: requests.post(
                    f"{self.api_config['base_url']}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=15
                )
            )

            if response.status_code == 200:
                result = response.json()
                emotion_text = result['choices'][0]['message']['content'].strip()

                # 清理JSON格式
                cleaned_response = re.sub(r"^```(?:json)?\n(.*?)\n```$", r"\1", emotion_text, flags=re.DOTALL)

                try:
                    emotion_data = json.loads(cleaned_response)
                    return emotion_data
                except json.JSONDecodeError:
                    return self._simple_emotion_analysis(text)
            else:
                return self._simple_emotion_analysis(text)

        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return self._simple_emotion_analysis(text)

    def _simple_emotion_analysis(self, text: str) -> dict:
        """简单情感分析"""
        text_lower = text.lower()

        emotion_keywords = {
            'joy': ['开心', '高兴', '快乐', '愉快', 'happy', 'joy', 'good'],
            'sadness': ['难过', '伤心', '悲伤', 'sad', 'cry', 'hurt'],
            'anger': ['生气', '愤怒', '恼火', 'angry', 'mad', 'hate'],
            'fear': ['害怕', '恐惧', '担心', 'scared', 'fear', 'worried'],
            'surprise': ['惊讶', '震惊', '意外', 'wow', 'surprised', 'amazing'],
            'disgust': ['恶心', '厌恶', '反感', 'disgusting', 'gross', 'yuck']
        }

        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = matches / len(keywords) if keywords else 0

        if all(score == 0 for score in emotion_scores.values()):
            primary_emotion = 'neutral'
            confidence = 1.0
        else:
            primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
            confidence = min(emotion_scores[primary_emotion] * 10, 1.0)

        emotion_scores['neutral'] = 1.0 - confidence if confidence < 1.0 else 0.0

        return {
            'primary_emotion': primary_emotion,
            'confidence': confidence,
            'all_emotions': emotion_scores
        }

    def generate_flame_expression(self, emotion_data: dict) -> list:
        """生成FLAME表情参数 - 使用加权方式"""
        all_emotions = emotion_data.get('all_emotions', {})

        # 初始化结果表情参数
        weighted_expression = np.zeros(50)

        # 对所有情感进行加权计算
        for emotion, weight in all_emotions.items():
            if emotion in self.emotion_to_expression and weight > 0:
                emotion_expression = self.emotion_to_expression[emotion]
                weighted_expression += weight * emotion_expression

        # 如果没有任何情感权重，使用neutral
        if np.sum(np.abs(weighted_expression)) == 0:
            weighted_expression = self.emotion_to_expression['neutral']

        # 限制参数范围
        weighted_expression = np.clip(weighted_expression, -2.0, 2.0)
        return weighted_expression.tolist()

    def generate_flame_pose(self, emotion_data: dict) -> list:
        """生成FLAME姿态参数 - 使用加权方式"""
        all_emotions = emotion_data.get('all_emotions', {})

        # 初始化结果姿态参数
        weighted_pose = np.zeros(6)

        # 对所有情感进行加权计算
        for emotion, weight in all_emotions.items():
            if emotion in self.emotion_to_pose and weight > 0:
                emotion_pose = self.emotion_to_pose[emotion]
                weighted_pose += weight * emotion_pose

        # 如果没有任何情感权重，使用neutral
        if np.sum(np.abs(weighted_pose)) == 0:
            weighted_pose = self.emotion_to_pose['neutral']

        # 限制参数范围
        weighted_pose = np.clip(weighted_pose, -1.0, 1.0)
        return weighted_pose.tolist()

    async def call_cosyvoice_api(self, text: str, emotion: str = 'neutral') -> bytes:
        """异步调用CosyVoice API生成语音"""
        try:
            emotion_prompt = self.emotion_to_voice_prompt.get(emotion, self.emotion_to_voice_prompt['neutral'])
            voice_input = f"{emotion_prompt} <|endofprompt|>{text}"

            language = self.detect_language(text)
            voice_model = "FunAudioLLM/CosyVoice2-0.5B:alex" if language == 'english' else "FunAudioLLM/CosyVoice2-0.5B:diana"

            url = "https://api.siliconflow.cn/v1/audio/speech"
            payload = {
                "model": "FunAudioLLM/CosyVoice2-0.5B",
                "input": voice_input,
                "voice": voice_model,
                "response_format": "mp3",
                "sample_rate": 32000,
                "stream": False,
                "speed": 1,
                "gain": 0
            }

            headers = {
                "Authorization": f"Bearer {self.api_config['api_key']}",
                "Content-Type": "application/json"
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: requests.post(url, json=payload, headers=headers, timeout=30)
            )

            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"CosyVoice API调用失败: {response.status_code}")

        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            raise


# 初始化API实例
voice_api = VoiceEmotionAPI()

# 存储生成的音频文件
audio_storage = {}


@app.post("/analyze_voice", response_model=AnalysisResponse)
async def analyze_voice(audio: UploadFile = File(...)):
    """主要API接口：分析语音并返回所有参数"""
    try:
        # 检查文件类型
        if not audio.filename:
            raise HTTPException(status_code=400, detail="没有选择文件")

        # 保存上传的音频文件
        file_extension = os.path.splitext(audio.filename)[1]
        temp_filename = f"upload_{uuid.uuid4().hex}{file_extension}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)

        async with aiofiles.open(temp_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)

        try:
            # 1. 语音转文本 (在线程池中执行)
            loop = asyncio.get_event_loop()
            recognized_text = await voice_api.speech_to_text(temp_path)

            # 2. 生成回复
            response_text = await voice_api.call_llm_api(recognized_text)

            # 3. 情感分析
            emotion_data = await voice_api.analyze_emotion(response_text)

            # 4. 生成表情参数和姿态参数
            expression_params = voice_api.generate_flame_expression(emotion_data)
            pose_params = voice_api.generate_flame_pose(emotion_data)

            # 5. 生成回复语音
            try:
                audio_data = await voice_api.call_cosyvoice_api(response_text, emotion_data['primary_emotion'])

                # 保存生成的音频
                audio_id = str(uuid.uuid4())
                audio_filename = f"response_{audio_id}.mp3"

                # 获取当前Python文件所在目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # 创建result目录路径
                result_dir = os.path.join(current_dir, "result")
                # 如果result目录不存在，则创建它
                os.makedirs(result_dir, exist_ok=True)
                # 设置音频文件的完整路径
                audio_path = os.path.join(result_dir, audio_filename)

                async with aiofiles.open(audio_path, 'wb') as f:
                    await f.write(audio_data)

                # 存储音频信息
                audio_storage[audio_id] = {
                    'path': audio_path,
                    'filename': audio_filename
                }

                has_audio = True

            except Exception as e:
                logger.error(f"语音生成失败: {e}")
                audio_id = None
                has_audio = False

            # 构建响应
            result = AnalysisResponse(
                success=True,
                input_text=recognized_text,
                response_text=response_text,
                emotion_analysis=EmotionAnalysis(**emotion_data),
                expression_parameters=expression_params,
                pose_parameters=pose_params,
                audio_generated=has_audio,
                audio_id=audio_id if has_audio else None
            )

            return result

        finally:
            # 清理上传的临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/get_audio/{audio_id}")
async def get_audio(audio_id: str):
    """获取生成的音频文件"""
    if audio_id not in audio_storage:
        raise HTTPException(status_code=404, detail="音频文件不存在")

    audio_info = audio_storage[audio_id]

    if not os.path.exists(audio_info['path']):
        raise HTTPException(status_code=404, detail="音频文件已过期")

    return FileResponse(
        audio_info['path'],
        media_type='audio/mpeg',
        filename=audio_info['filename']
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        message="Voice Emotion API is running"
    )


@app.get("/", response_model=APIInfo)
async def get_api_info():
    """API文档"""
    return APIInfo(
        message="Voice Emotion Analysis API",
        endpoints={
            "POST /analyze_voice": "上传音频文件进行分析",
            "GET /get_audio/{audio_id}": "下载生成的音频文件",
            "GET /health": "健康检查",
            "GET /docs": "Swagger API文档",
            "GET /redoc": "ReDoc API文档"
        },
        usage={
            "analyze_voice": {
                "method": "POST",
                "content_type": "multipart/form-data",
                "parameters": {
                    "audio": "audio file (wav, mp3, etc.)"
                },
                "response": {
                    "input_text": "recognized text from audio",
                    "response_text": "AI generated response",
                    "emotion_analysis": "emotion analysis result",
                    "expression_parameters": "FLAME expression parameters (50 values)",
                    "pose_parameters": "FLAME pose parameters (6 values)",
                    "audio_generated": "whether audio was generated successfully",
                    "audio_id": "ID to download generated audio"
                }
            }
        }
    )


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8004)