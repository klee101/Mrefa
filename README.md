# 第 9 组 混合现实具身脸部动画 代码说明文档

本项目旨在实现混合现实具身脸部动画，采用现实人脸图像作为输入，并参考 DECA 方法，利用 FLAME 模型实现参数化三维人脸重建，提取并映射人脸的形状、表情、位姿参数。对于生成的人脸模型，系统通过 AR 头显设备进行语音交互，实时采集用户语音。借助大语言模型（LLM）生成回复，同时进行语义解析与情感识别，驱动人脸模型生成符合语义和情感的动态表情。同时，结合语义信息合成具有情感语调的 AI 语音，实现更自然的语音响应。项目代码分为以下 4 个模块：

1. **基于 DECA 的使用现实图像得到参数化模型模块**
2. **基于 FLAME 参数化动态建模模块**
3. **语音情感分析模块**
4. **Unity 集成 MR 模块**

结尾将介绍 `MREFA-v1.0.apk` 部署应用的注意事项。

---

## 1. 基于 DECA 的使用现实图像得到参数化模型模块

### 模块概述

该模块基于 DECA（Detailed Expression Capture and Animation）技术，从输入的现实人脸图像中提取基于 FLAME 模型的参数化表达。

### 主要功能

- **从图像提取 FLAME 形状参数**: 用于固定参数化模型的形状与现实人脸图像相符合
- **从图像提取 FLAME 表情参数**: 初始化参数化模型的表情
- **从图像提取 FLAME 姿态参数**: 初始化参数化模型的姿态
- **从图像提取 FLAME 纹理参数**: 用于后续可能的人脸纹理映射（但由于 DECA 方法纹理映射效果较差故未使用）

### 技术栈

- **DECA 预训练模型**
- **FLAME 参数化建模模型**
- **PyTorch**
- **FastAPI**

### FastAPI 路由接口

#### POST `/deca-flame-params/`

**主要 API 接口，完整的基于现实人脸图像提取基于 FLAME 模型的参数化表达流程**

**输入参数**:

- `file`: 上传的现实人脸图像文件

**处理流程**:

1. 加载图像
2. 使用 `datasets.TestData` 对图像进行裁剪和预处理，并检测人脸区域
3. 使用 DECA 模型对图像进行编码，提取出 FLAME 参数
4. 将模型输出的参数从 PyTorch 张量转换为 NumPy 数组，并进一步转换为 Python 的列表格式
5. 返回一个 JSON 对象，包含提取的 FLAME 参数

**返回数据**:

```json
{
  "shape_params": [浮点数列表],  # 人脸的形状参数
  "expression_params": [浮点数列表],  # 人脸的表情参数
  "pose_params": [浮点数列表],  # 人脸的姿态参数
  "tex_params": [浮点数列表]  # 人脸的纹理参数
}
```

### 使用方法

#### 1.环境配置

环境配置的方法参考技术文档（by 连炜乐）链接：

https://foxions.com/ac/repr/siggraph/deca-repr/

#### 2.启动服务

在 `service1-deca` 文件夹下，使用 `uvicorn` 命令

```bash
uvicorn app:app --host 0.0.0.0 --port 8002
```

### API 调用示例

#### cURL 调用示例

```bash
curl -X POST http://localhost:8002/deca-flame-params/ \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
```

## 2. 基于 FLAME 参数化动态建模模块

### 模块概述

该模块使用 FLAME 参数化模型，通过输入的参数动态生成三维人脸网格模型，支持单次网格生成和参数插值生成多个动态网格。

### 主要功能

- **生成单个网格模型**: 接收 FLAME 的形状、表情和姿态参数，生成对应的三维人脸网格。
- **生成参数插值的动态网格模型**: 在给定的起始和目标参数之间进行插值，生成多帧动态变化的网格模型。

### 技术栈

- **FLAME 参数化建模模型**
- **PyTorch**
- **FastAPI**

### FastAPI 路由接口

#### POST `/generate_mesh/`

**单个网格生成接口**

**输入参数**:

- `shape_params` (List[float]): FLAME 形状参数，长度必须为 `config.shape_params`。
- `expression_params` (List[float]): FLAME 表情参数，长度必须为 `config.expression_params`。
- `pose_params` (List[float]): FLAME 姿态参数，长度必须为 `config.pose_params`。

**处理流程**:

1. 验证输入参数的长度。
2. 将参数转换为 PyTorch 张量并送入 FLAME 模型。
3. 从模型中获取网格顶点。
4. 返回网格顶点和面片索引。

**返回数据**:

```json
{
  "vertices": [[x, y, z], ...],  # 网格顶点坐标列表
  "faces": [[v1, v2, v3], ...]  # 网格面片索引列表
}
```

#### POST `/generate_interpolated_meshes/`

**动态插值网格生成接口**

**输入参数**:

- `old_params` (FlameParams): 起始参数，包括形状、表情和姿态。
- `new_params` (FlameParams): 目标参数，包括形状、表情和姿态。
- `num_interpolations` (int): 插值帧数。

**处理流程**:

1. 验证起始和目标参数的长度，以及插值帧数是否大于零。
2. 在起始和目标参数之间进行线性插值，生成多个插值参数。
3. 使用 FLAME 模型生成每个插值参数对应的网格。
4. 返回所有插值网格的顶点和面片索引。

**返回数据**:

```json
[
  {
    "vertices": [[x, y, z], ...],  # 插值网格的顶点坐标列表
    "faces": [[v1, v2, v3], ...]  # 插值网格的面片索引列表
  },
  ...
]
```

### 使用方法

#### 1.环境配置

环境配置的方法参考技术文档（by 连炜乐）链接：

https://foxions.com/ac/repr/siggraph/flame/

#### 2.启动服务

在 `service2-flame` 文件夹下，使用 `uvicorn` 命令

```bash
uvicorn app:app --host 0.0.0.0 --port 8003
```

### API 调用示例

#### cURL 调用示例

cURL 调用 `/generate_mesh/`

```bash
curl -X POST http://localhost:8003/generate_mesh/ \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
           "shape_params": [0.0, 0.1, ...],
           "expression_params": [0.0, 0.2, ...],
           "pose_params": [0.0, 0.3, ...]
         }'
```

cURL 调用 `/generate_mesh/`

```bash
curl -X POST http://localhost:8003/generate_interpolated_meshes/ \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
           "old_params": {
             "shape_params": [0.0, 0.1, ...],
             "expression_params": [0.0, 0.2, ...],
             "pose_params": [0.0, 0.3, ...]
           },
           "new_params": {
             "shape_params": [0.5, 0.6, ...],
             "expression_params": [0.4, 0.5, ...],
             "pose_params": [0.7, 0.8, ...]
           },
           "num_interpolations": 5
         }'
```

---

## 3. 语音情感分析模块

### 模块概述

该模块是一个基于 FastAPI 开发的语音情感分析系统，集成了语音识别、情感分析、FLAME 参数生成和语音合成功能。系统能够接收用户的语音输入，识别语音内容，生成 AI 回复，分析情感状态，并生成相应的 3D 面部表情和姿态参数。

### 主要功能

- **语音转文本（STT）**: 将用户语音转化为文本内容
- **智能对话回复生成**: 使用大语言模型生成自然语言回复
- **情感分析**: 分析文本内容的情感状态及其权重，如喜悦、愤怒、悲伤等
- **FLAME 表情参数生成**: 根据情感分析结果生成对应的 FLAME 表情参数
- **FLAME 姿态参数生成**: 根据语音内容生成对应的 FLAME 姿态参数
- **文本转语音（TTS）**: 将 AI 的回复合成为语音输出

### 技术栈

- **异步处理**: asyncio, aiofiles
- **语音识别**: SiliconFlow API (SenseVoiceSmall 模型)
- **大语言模型**: SiliconFlow API (DeepSeek-V3 模型)
- **语音合成**: SiliconFlow API (CosyVoice2-0.5B 模型)
- **HTTP 请求**: requests
- **并发处理**: ThreadPoolExecutor

### FastAPI 路由接口

#### POST `/analyze_voice`

**主要 API 接口，完整的语音分析流程**

**输入参数**:

- `audio`: 上传的音频文件（支持 wav, mp3 等格式）

**处理流程**:

1. 接收并保存上传的音频文件
2. 调用语音识别 API 转换为文本
3. 使用大语言模型生成回复
4. 分析回复文本的情感状态
5. 生成 FLAME 表情和姿态参数
6. 合成带情感的回复语音
7. 返回完整的分析结果

**返回数据**:

```json
{
  "success": true,
  "input_text": "用户说的话",
  "response_text": "AI 的回复",
  "emotion_analysis": {
    "primary_emotion": "joy",
    "confidence": 0.85,
    "all_emotions": {...}
  },
  "expression_parameters": [50个浮点数],
  "pose_parameters": [6个浮点数],
  "audio_generated": true,
  "audio_id": "音频文件 ID"
}
```

#### GET `/get_audio/{audio_id}`

**下载生成的音频文件**

**功能**: 根据 audio_id 下载对应的音频文件
**返回**: MP3 格式的音频流

#### GET `/health`

**健康检查接口**

**功能**: 检查 API 服务状态
**返回**: 服务运行状态信息

#### GET `/`

**API 信息接口**

**功能**: 提供 API 使用说明和接口文档
**返回**: 详细的 API 使用指南

### 使用方法

#### 1. 环境配置

安装依赖

```bash
pip install fastapi uvicorn aiofiles numpy requests SpeechRecognition pydantic
```

#### 2. 启动服务

直接运行

```bash
python main.py
```

使用 uvicorn 命令

```bash
uvicorn main:app --host 0.0.0.0 --port 8004 --reload
```

### API 调用示例

#### Python 调用示例

```python
import requests

# 上传音频文件进行分析
url = "http://localhost:8004/analyze_voice"
with open("test_audio.wav", "rb") as f:
    files = {"audio": f}
    response = requests.post(url, files=files)
    result = response.json()
    print(result)

# 下载生成的音频
if result["audio_generated"]:
    audio_url = f"http://localhost:8004/get_audio/{result['audio_id']}"
    audio_response = requests.get(audio_url)
    with open("response_audio.mp3", "wb") as f:
        f.write(audio_response.content)
```

#### cURL 调用示例

```bash
# 上传音频分析
curl -X POST "http://localhost:8004/analyze_voice" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "audio=@test_audio.wav"

# 下载生成的音频
curl -X GET "http://localhost:8004/get_audio/{audio_id}" \
     --output response_audio.mp3
```

### 配置说明

#### API 密钥配置

```python
self.api_config = {
    'type': 'openai',
    'base_url': 'https://api.siliconflow.cn/v1',
    'api_key': 'your_api_key_here',  # 需要替换为实际的API密钥
    'model': 'deepseek-ai/DeepSeek-V3'
}
```

#### 系统提示词配置

```python
self.system_prompt = """你是一个友好、有情感的AI助手。请遵循以下规则：
1. 用自然、温暖的语调回复
2. 回复要简洁，控制在30字以内
3. 根据用户的情感状态调整回复风格
4. 支持中文和英文对话
5. 输出格式要求：仅返回纯文本对话内容"""
```

### 常见问题

#### Q1: 如何更换 API 服务商？

A: 修改`api_config`中的配置，并相应调整 API 调用方法。

#### Q2: 如何添加新的情感类型？

A: 在`emotion_to_expression`和`emotion_to_pose`中添加新的映射关系。

#### Q3: 如何调整 FLAME 参数的生成策略？

A: 修改`generate_flame_expression`和`generate_flame_pose`方法中的加权算法。

#### Q4: 如何支持更多语言？

A: 在`detect_language`方法中添加新的语言检测逻辑，并在语音合成中添加对应的语音模型。

## 4. Unity 集成 MR 模块

### 模块概述

该模块通过 Unity 实现混合现实具身脸部动画的实时渲染和交互。将前面三个模块生成的 FLAME 参数化模型导入 Unity，并结合语音情感分析结果动态更新面部动画，最终结合燧光 Rhino X Pro 眼镜实现 MR 的混合现实应用部署。

### 主要功能

- **从图像提取 FLAME 形状参数**: 用于固定参数化模型的形状与现实人脸图像相符合
- **从图像提取 FLAME 表情参数**: 初始化参数化模型的表情
- **从图像提取 FLAME 姿态参数**: 初始化参数化模型的姿态
- **从图像提取 FLAME 纹理参数**: 用于后续可能的人脸纹理映射（但由于 DECA 方法纹理映射效果较差故未使用）

### 技术栈

- **Unity**
- **C#脚本**
- **HTTP 请求**
- **JSON 数据解析**

### 使用方法

#### 1. 环境配置

基于 `2021.3.9f1c1` 版本的 Unity 引擎

燧光眼镜开发环境配置的方法参考技术文档链接：

https://doc.ximmerse.com/sdkconf/unityxrsdk/index.html

配置好开发环境后导入 `MREFA-lwl_hpf_gyc.unitypackage` 包即可

#### 2. 使用说明

导入包后，`Scenes` 文件夹中 `MREFA` 的 Scene 即我们开发的应用场景。

### 代码介绍

`Scripts` 文件夹中含有开发使用到的 C#脚本

1. AudioUtils：用于转化 Unity 内置录制的 AudioClip 至 Wav 以用于 JSON 发送请求；
2. CameraAccess：用于获取 Rhino X Pro 眼镜中的内置摄像头，以及 UI 界面上所有与摄像头相关的组件的控制；
3. FLAMEMeshUpdater：用于通过与后端交互生成和更新基于 FLAME 模型的 3D 面部模型，支持从图片提取面部参数、生成网格、表情动画过渡、语音分析与情感识别，以及同步播放 AI 生成的语音和动画：
   1. PostImageAndFetchParams(Texture2D inputImage)：发送图片到后端，获取 FLAME 参数；
   2. PostMeshRequest()：发送 FLAME 参数到后端，生成 3D 网格数据；
   3. UpdateMesh(float[][] vertices, int[][] faces)：将后端返回的网格数据更新到 Unity 中；
   4. PostAnimateMeshTransition()：根据旧参数和新参数，生成多个插值网格，用于动画过渡；
   5. PlayAnimation()：逐步更新网格并播放动画；
   6. PostAnalyzeVoiceRequest()：上传用户语音文件到后端进行分析，获取情感、表情参数等信息；
   7. DownloadAudio(string audioID)：下载后端生成的 AI 语音文件；
   8. ReplayTalk()：回放 AI 语音和面部动画。
4. ImageLoader：用于加载示例图片并提供选择；
5. Microphone Access：用于获取 Rhino X Pro 眼镜中的内置麦克风，以及 UI 界面上所有与麦克风相关的组件的控制；
6. Reposition：用于调整 UI 界面方向和位置；
7. SetUp：用于获取 Rhino X Pro 眼镜中的内置摄像头和麦克风的权限。

## `MREFA-v1.0.apk` 部署应用的注意事项

该 apk 应用需要通过 `adb` 命令安装至 Rhino X Pro 眼镜中，具体方式参考文章：https://doc.ximmerse.com/sdkconf/unityxrsdk/index.html。

由于项目组测试应用时使用的是本地部署服务，通过反向通道实现公网 IP 的服务访问。因此，如需要对项目进行测试请联系组长——胡鹏飞同学部署服务，即可进行测试。
