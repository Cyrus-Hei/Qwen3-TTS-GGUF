"""
44-Test-Audio-Recovery.py - 演示音频恢复机制
展示如何从仅包含特征的 JSON 中，通过注入解调器恢复音频并播放。
"""
import os
import numpy as np
import sounddevice as sd
from qwen3_tts_gguf.engine import TTSEngine
from qwen3_tts_gguf.result import TTSResult

def main():
    # 0. 初始化引擎
    engine = TTSEngine(model_dir="model-base")
    
    # 1. 模拟一个“纯特征”的 JSON 存档
    # 实际场景中，这可能是从服务器下载或从数据库读取的
    JSON_PATH = "output/test_anchor_no_audio.json"
    
    # 首先生成一个锚点用于测试
    print("\n1️⃣ 正在生成并保存一个不含音频的 JSON 锚点...")
    text = "这只是一个用于测试音频恢复的参考句。"
    # 构造一个模拟的 TTSResult
    mock_res = TTSResult(
        text=text,
        spk_emb=np.zeros(2048, dtype=np.float32),
        text_ids=[1, 2, 3],
        codes=np.zeros((50, 16), dtype=np.int64), # 50 帧全 0 (静音码)
    )
    # 保存时故意不包含音频
    mock_res.save_json(JSON_PATH, include_audio=False)
    
    # 2. 重新加载
    print(f"\n2️⃣ 从 {JSON_PATH} 重新加载结果...")
    reloaded_res = TTSResult.from_json(JSON_PATH)
    
    # 3. 尝试播放 (预期触发友好警告)
    print("\n3️⃣ 尝试直接播放加载的结果:")
    reloaded_res.play()
    
    # 4. 注入引擎的解码器进行“复活”
    print("\n4️⃣ 正在执行注入式解码 (.decode(engine.decoder))...")
    # engine.decoder 遵循 decoder 接口 (具备 .decode 方法)
    reloaded_res.decode(engine.decoder)
    
    # 5. 再次尝试播放 (预期成功)
    if reloaded_res.audio is not None:
        print("\n5️⃣ 渲染成功！现在的音频长度:", len(reloaded_res.audio))
        print("🔊 正在播放 (虽然码是 0，但架构已通)...")
        reloaded_res.play(blocking=True)
    else:
        print("\n❌ 渲染失败。")

    engine.shutdown()

if __name__ == "__main__":
    main()
