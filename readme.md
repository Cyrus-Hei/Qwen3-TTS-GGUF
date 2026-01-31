# Qwen3-TTS GGUF

用 llama.cpp 跑的 Qwen3-TTS，支持流式合成、声音克隆。

## 模型类型

本项目支持三种官方模型，对应三种场景：

| 源模型 | 场景 |
| :--- | :--- |
| Qwen3-TTS-12Hz-1.7B-Base | 零样本声音克隆 |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 内置精品音色 + 风格指令 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 用自然语言设计音色 |

你想用哪个，就导出哪个，但是需要先下载官方模型才能导出。

## 性能表现

在我的 RTX 5050 上的实测数据：

- **RTX 5050 (独显)**: RTF 0.5 (实时率，1秒音频只需0.5秒生成)
- **CPU**: RTF 2.8
- **集显**: RTF 1.6

RTF < 1 就表示生成速度比实时播放还快。


## 快速开始

#### 下载模型

- [Qwen3-TTS-12Hz-1.7B-Base](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)


#### 配置路径

打开 `export_config.py`，改两行：

```python
MODEL_DIR = "官方模型路径"   # 比如 ~/.cache/.../Qwen3-TTS-12Hz-1.7B-Base
EXPORT_DIR = "./model-base"  # 输出目录
```

#### 阶段一：导出小组件

```bash
python 11-Export-Codec-Encoder.py   # 编码器，克隆用
python 12-Export-Speaker-Encoder.py # 说话人特征提取器
python 13-Export-Decoder.py          # 解码器，把 codes 变声音
python 14-Export-Embeddings.py       # Embedding 权重
python 15-Copy-Tokenizer.py          # 文本分词器
```

#### 阶段二：导出大师（Talker）

大师是 1.42B 的 LLM backbone，负责理解文本、生成语音骨架：

```bash
python 21-Extract-Talker-Weights.py    # 拆权重
python 22-Prepare-Talker-Tokenizer.py  # 造迷你词表（GGUF 需要）
python 23-Convert-Talker-GGUF.py       # 转 GGUF
```

#### 阶段三：导出工匠（Predictor）

工匠是 142M 的小模型，负责给骨架补充血肉：

```bash
python 31-Extract-Predictor-Weights.py
python 32-Prepare-Predictor-Tokenizer.py
python 33-Convert-Predictor-GGUF.py
```

导完之后，`EXPORT_DIR` 里就有你需要的所有文件了。

## 推理

### 交互模式（推荐）

```bash
python 51-Interactive-Clone.py
```

启动后直接打字，边推边播。

### 脚本模式

三个示例脚本，对应三种模型：

```bash
python 41-Inference-Base.py    # 声音克隆
python 42-Inference-Custom.py  # 精品音色
python 43-Inference-Design.py  # 音色设计
```

## 代码调用

```python
from qwen3_tts_gguf import TTSEngine, TTSConfig

engine = TTSEngine(model_dir="model-base")
stream = engine.create_stream()

# 从 JSON 加载音色（无损）
stream.set_voice("output/sample.json")

# 流式合成
result = stream.clone("你好，世界！", streaming=True)
stream.join()  # 等播完

# 保存
result.save("output.wav")
result.save("output.json")  # 保存 codes，下次可以无损加载
```

## 可用说话人

CustomVoice 模型内置 9 个音色：

| ID | 说明 |
| :--- | :--- |
| vivian | 年轻女声，明亮利落 |
| serena | 温暖女声，柔和亲切 |
| uncle_fu | 成熟男声，沉稳低沉 |
| dylan | 北京男声，自然清晰 |
| eric | 成都男声，略带沙哑 |
| ryan | 活力男声，节奏感强 |
| aiden | 阳光美男，中频清澈 |
| ono_anna | 日语女声，俏皮轻快 |
| sohee | 韩语女声，温润动情 |

## 支持的语言

- chinese, english, japanese, korean
- german, spanish, french, russian, italian, portuguese
- beijing_dialect, sichuan_dialect

## 架构简述

用人体器官来理解：

1. **耳朵**（Encoder）: 听参考音频，提取音色特征
2. **大脑**（Talker）: 生成语音骨架 (28层, 1.42B)
3. **双手**（Predictor）: 补充细节 (8层, 142M)
4. **嘴巴**（Decoder）: 把 codes 解码成声音

推理引擎：
- Talker / Predictor: llama.cpp (GGUF)
- Encoder / Decoder: ONNX Runtime

## 常见问题

**Q: 为什么不用官方 PyTorch？**

官方实现要大显存。llama.cpp 省资源，还能用 Vulkan 加速。

**Q: 流式和离线有什么区别？**

流式边推边播，首包延迟低。离线推完再播，适合批量生成。

**Q: 怎么调质量？**

`TTSConfig(temperature=0.8)` 控制随机性。低了呆板，高了飘忽，推荐 0.7-1.0。

## 相关链接

- [Qwen3-TTS 技术报告](./Qwen3-TTS%20Technical%20Report.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
