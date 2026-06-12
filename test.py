"""
51-Interactive-Clone.py - 交互式流式语音合成终端 (V3 架构版)
基于最新的 DecoderProxy 和 TTSStream 架构。
"""
import os
import sys
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult
from qwen3_tts_gguf.inference.schema.constants import SPEAKER_MAP, LANGUAGE_MAP

def interactive_session():
    print("\n🚀 正在启动 Qwen3-TTS 交互式终端...")
    
    # 1. 引擎初始化
    engine = TTSEngine('model-base', verbose=False)
    stream = engine.create_stream()

    if stream is None:
        print("❌ 引擎初始化失败，请检查模型文件路径是否正确。")
        return

    # 2. 默认配置
    cfg = TTSConfig(temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45)
    last_result: Optional[TTSResult] = None
    REF_AUDIO = "path_to_audio_file" # Put path to reference autio file(voice to clone) here
    REF_TEXT = "transcript_of_audio_file" # Put transcript of reference autio here(not file path, text only)
    print(f"Loading reference voice profile...")
    last_result = stream.set_voice(REF_AUDIO, REF_TEXT)

    print("\n✨ 引擎就绪！您可以直接输入文本进行合成，或输入 /help 查看指令。")

    try:
        while True:
            try:
                raw_input = input("\n[Qwen3] >>> ").strip()
            except EOFError:
                break
                
            if not raw_input:
                continue

            # --- 标准合成 (Clone 模式) ---
            if not stream.voice:
                print("⚠️  voice not set! ")
                continue

            try:
                print("🎤 正在流式合成...")
                last_result = stream.clone(raw_input, zero_shot=False, config=cfg)
                stream.join()
                if last_result:
                    print(f"✨ 完成! [RTF: {last_result.rtf:.2f}]")
            except RuntimeError as e:
                print(f"💡 提示: {e}")
                print("   ⚠️ voice not set! ")

    except KeyboardInterrupt:
        print("\n👋 退出会话。")
    except Exception as e:
        print(f"\n⚠️ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.shutdown()

if __name__ == "__main__":
    interactive_session()