"""
103-Test-Wav-Streaming.py - 验证后台音频流式推送
"""
import os
import sys
import time
import numpy as np
import soundfile as sf

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.engine import TTSEngine

def test_wav_streaming():
    print("\n" + "="*50)
    print("🧪 测试: 后台流式音频推送 (External WAV)")
    print("="*50)

    WAV_PATH = "output/hybrid.wav"
    if not os.path.exists(WAV_PATH):
        print(f"❌ 找不到测试音频: {WAV_PATH}")
        return

    # 1. 启动引擎（这会拉起后台播放进程）
    engine = TTSEngine()
    
    print("⏳ 正在等待后台工作进程就绪 (首次启动可能需要几秒)...")
    if engine.mouth.wait_until_ready(timeout=10):
        print("✅ 后台进程已全面就绪！")
    else:
        print("⚠️ 等待就绪超时，尝试继续执行...")
    # 2. 读取音频文件
    print(f"📖 正在读取音频: {WAV_PATH}")
    audio, sr = sf.read(WAV_PATH)
    if sr != 24000:
        print(f"⚠️ 注意: 音频采样率是 {sr}Hz，系统期望 24000Hz (播放速度可能会变化)")
        
    # 3. 模拟流式推送
    # 每次推送 12000 个采样点 (约 0.5 秒)
    chunk_size = 12000 
    print(f"\n🎧 正在逐块推送音频到后台播放进程 (Chunk Size: {chunk_size})...")
    
    total_len = len(audio)
    for i in range(0, total_len, chunk_size):
        chunk = audio[i : i + chunk_size].astype(np.float32)
        print(f"   ▶️ 推送 Chunk {i//chunk_size + 1} ({len(chunk)} samples)...")
        
        # 使用新增加的 raw_play 接口
        engine.mouth.raw_play(chunk)
        
        # 模拟产生音频的延时
        time.sleep(0.4) 
        
    print("\n✅ 所有音频块已推送完毕。")
    print("⏳ 正在等待后台播放结束 (5s)...")
    time.sleep(5)
    
    # 4. 彻底关闭
    engine.shutdown()
    print("\n🏁 测试完成。")

if __name__ == "__main__":
    test_wav_streaming()
