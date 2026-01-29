from qwen3_tts_gguf.codec_export import StatefulCodecExportWrapper
from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model
import torch

try:
    model_path = "./Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model = Qwen3TTSTokenizerV2Model.from_pretrained(model_path)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
