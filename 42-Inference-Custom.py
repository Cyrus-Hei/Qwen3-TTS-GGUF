"""
42-Inference-Custom.py - Qwen3-TTS 精品音色推理脚本 (Engine 版)
"""
import os
import soundfile as sf
import sounddevice as sd
from qwen3_tts_gguf.engine import TTSEngine

def main():
    # 1. 配置参数
    # 官方支持音色: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
    TARGET_TEXT = "你好，我是千问3-TTS，你今天过得好吗？"
    SPEAKER = "Vivian"
    INSTRUCT = "用温柔的语气说"
    
    # 2. 初始化引擎
    print(f"🚀 [Custom-Inference] 正在初始化引擎")
    # 注意：如果您的定制模型放在 model-custom 目录，请指定
    engine = TTSEngine(model_dir="model-custom")
    stream = engine.create_stream()

    # 3. 进行推理 (调用 custom 模式)
    print(f"🎭 正在合成 (指令: {INSTRUCT})...")
    result = stream.custom(
        text=TARGET_TEXT,
        speaker=SPEAKER,
        instruct=INSTRUCT,
        verbose=True
    )
    
    # 4. 播放与保存
    if result.audio is not None:
        print(f"\n✅ 合成成功！ RTF: {result.rtf:.2f}")
        sf.write("output_custom.wav", result.audio, 24000)
        
        print("🔊 正在播放...")
        sd.play(result.audio, 24000)
        sd.wait()
    else:
        print("❌ 合成失败。")

    engine.shutdown()

if __name__ == "__main__":
    main()
