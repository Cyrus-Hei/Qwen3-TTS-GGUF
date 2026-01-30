"""
97-Test-Chat-Manager.py
验证 ChatManager 的精确删除与 GC 逻辑。
"""
import os
import time
from qwen3_tts_gguf import TTSEngine

def test_memory_gc():
    print("\n" + "="*50)
    print("🧪 测试: ChatManager 记忆自动清理 (GC) 验证")
    print("="*50)
    
    # 初始化引擎
    engine = TTSEngine(model_dir="model")
    
    # 创建负载较小的 Stream (n_ctx=512)，方便触发 GC
    stream = engine.create_stream(speaker_id="vivian", language="chinese", n_ctx=4096)
    
    sentences = [
        "你好，我是千问，你今天过得好吗？",
        "嗯，我知道了，下去吧。",
        "太可恶了！！！",
        "这是第四轮",
        "这是第五轮",
        "这是第六轮"
    ]
    
    for i, text in enumerate(sentences):
        print(f"\n--- [Round {i+1}] 正在合成: {text} ---")
        try:
            res = stream.synthesize(text, play=True, verbose=True, temperature=0.7)
            res.print_stats()
            print(f"当前物理位置 (cur_pos): {stream.master.cur_pos}")
            print(f"基准记忆状态: {stream.chat.base_turn}")
        except Exception as e:
            print(f"❌ 轮次 {i+1} 失败: {e}")
            break
            
    engine.shutdown()

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    try:
        test_memory_gc()
        print("\n✅ GC 测试流程运行完毕。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
