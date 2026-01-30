"""
97-Test-Identity-Mode.py - 验证新版五要素 Identity 与 tts 接口
"""
import os
import sys
import numpy as np

# 确保能找到 qwen3_tts_gguf 包
sys.path.append(os.getcwd())

from qwen3_tts_gguf.engine import TTSEngine
from qwen3_tts_gguf.result import GenConfig

def test_identity_mode():
    print("\n" + "="*50)
    print("🧪 测试: 新版五要素 Identity 锚定与 tts 接口验证")
    print("="*50)

    # 1. 初始化引擎
    engine = TTSEngine(model_dir="model")
    
    # 2. 创建 Stream
    stream = engine.create_stream(n_ctx=4096)
    
    # 2.1 负面测试：未锚定直接 TTS 应该报错
    print("\n⚠️ [Negative Test] 尝试在未锚定身份的情况下调用 tts...")
    try:
        stream.tts("这应该会报错")
    except RuntimeError as e:
        print(f"✅ 捕获到预期的错误: {e}")
    
    # 3. 第一步：显式锚定音色 (Identity)
    print("\n🚀 [Step 1] 正在显式定调 (Set Identity)...")
    id_ref_text = "你好，我是千问，你今天过管好吗？"
    # 设置身份，获取定调时的合成结果
    res_id = stream.set_identity_from_speaker(speaker_id="vivian", text=id_ref_text, language="chinese")
    
    # 打印定调时的耗时统计
    print("\n📊 定调（锚定）性能报告:")
    res_id.print_stats()
    
    print(f"\n✅ Identity 状态: {'已锁定' if stream.identity.is_set else '未设置'}")
    if stream.identity.is_set:
        print(f"   ├─ 参考文字: '{stream.identity.text}'")
        print(f"   ├─ 参考文字长度: {len(stream.identity.text_ids)} tokens")
        print(f"   └─ 参考音频帧数: {stream.identity.codes.shape[0]} frames")

    # 4. 第二步：在锚定状态下进行多轮对话合成
    test_sentences = [
        "你好，今天天气真不错，我们出去走走吧？",
        "太可恶了！怎么会有这种事情发生！"
    ]
    
    for i, text in enumerate(test_sentences):
        print(f"\n--- [Round {i+1}] 正在合成: {text} ---")
        try:
            # 使用 GenConfig 传递推理参数
            cfg = GenConfig(temperature=0.7, max_steps=500)
            res = stream.tts(text, play=True, config=cfg)
            
            # 打印详细性能统计 (来自 res.stats)
            res.print_stats()
            
            # 验证全量结果返回 (TTSResult)
            print(f"🔍 结果核查 (TTSResult):")
            print(f"  ├─ 文本内容: {res.text}")
            print(f"  ├─ 文本 Token 数: {len(res.text_ids)}")
            print(f"  ├─ 音频 Codec 形状: {res.codes.shape}")
            print(f"  ├─ 全局音色向量形状: {res.spk_emb.shape}")
            print(f"  └─ 叠加特征帧数: {len(res.summed_embeds)}")
            
        except Exception as e:
            print(f"❌ 轮次 {i+1} 失败: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n✅ Identity 模式测试流程运行完毕。")

if __name__ == "__main__":
    test_identity_mode()
