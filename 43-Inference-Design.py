"""
43-Inference-Design.py - Qwen3-TTS 音色设计推理脚本 (Engine 版)
"""
import os
import soundfile as sf
import sounddevice as sd
from qwen3_tts_gguf.engine import TTSEngine

def main():
    # 1. 配置参数
    TARGET_TEXT = "这条语音由 VoiceDesign 模型生成。根据我的指令，你会听到一个非常有磁性的声音。"
    INSTRUCT = "具有磁性的中年男性声音，语速稍慢，语气沉稳。"
    
    # 2. 初始化引擎
    print(f"🚀 [Design-Inference] 正在初始化引擎 (Instruct: {INSTRUCT})...")
    # 注意：如果您的设计模型放在 model-design 目录，请指定
    engine = TTSEngine(model_dir="model-design")
    stream = engine.create_stream()

    # 3. 进行推理 (调用 design 模式)
    print(f"🎨 正在设计并合成...")
    result = stream.design(
        text=TARGET_TEXT,
        instruct=INSTRUCT,
        verbose=True
    )
    
    # 4. 播放与保存
    if result.audio is not None:
        print(f"\n✅ 合成成功！ RTF: {result.rtf:.2f}")
        sf.write("output_design.wav", result.audio, 24000)
        
        print("🔊 正在播放...")
        sd.play(result.audio, 24000)
        sd.wait()
    else:
        print("❌ 合成失败。")

    engine.shutdown()

if __name__ == "__main__":
    main()
