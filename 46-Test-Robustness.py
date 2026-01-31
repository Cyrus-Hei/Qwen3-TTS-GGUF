"""
46-Test-Robustness.py - 健壮性与防御性编程测试
"""
import os
import shutil
from pathlib import Path
from qwen3_tts_gguf.engine import TTSEngine

def test_engine_init():
    print("--- [Test] 引擎初始化防御检测 ---")
    # 1. 测试目录不存在
    engine_fake = TTSEngine(model_dir="non_existent_dir")
    if not engine_fake:
        print("✅ 拦截成功: 引擎在路径不存在时返回 False")
    else:
        print("❌ 拦截失败: 引擎未正确返回 False")

    # 2. 测试模型文件缺失 (模拟)
    os.makedirs("temp_model", exist_ok=True)
    with open("temp_model/tokenizer.json", "w") as f: f.write("{}") # 只放一个文件
    engine_missing = TTSEngine(model_dir="temp_model")
    if not engine_missing:
        print("✅ 拦截成功: 引擎在核心模型缺失时返回 False")
    shutil.rmtree("temp_model")

def test_voice_robustness():
    print("\n--- [Test] 音色设置鲁棒性检测 ---")
    engine = TTSEngine(model_dir="model-base")
    if not engine:
        print("❌ 跳过: model-base 不存在")
        return
    
    stream = engine.create_stream()
    
    # 1. 错误的文件路径
    success = stream.set_voice("non_existent.json")
    print(f"   传入非法 JSON 路径: {'成功' if success else '✅ 拦截成功'}")

    # 2. 传入不支持的格式
    success = stream.set_voice("README.md")
    print(f"   传入非法格式 (README): {'成功' if success else '✅ 拦截成功'}")

    # 3. 显存释放/切换测试
    print("\n--- [Test] 资源清理与切换测试 ---")
    for i in range(2):
        print(f"   切换测试 轮次 {i+1}...")
        temp_engine = TTSEngine(model_dir="model-base", verbose=False)
        if temp_engine:
            print(f"      引擎 {i+1} 就绪")
            temp_engine.shutdown()
            print(f"      引擎 {i+1} 已销毁")
    print("✅ 资源重分配测试完成 (未崩掉)")

if __name__ == "__main__":
    test_engine_init()
    test_voice_robustness()
    print("\n🎉 健壮性测试脚本运行结束。")
