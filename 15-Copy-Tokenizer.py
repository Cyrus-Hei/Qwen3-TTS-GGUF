"""
15-Copy-Tokenizer.py
复制文本 Tokenizer 到 model 文件夹，方便模型部署和管理。
"""
import os
import shutil
from pathlib import Path

from export_config import MODEL_DIR, EXPORT_DIR

def main():
    # 1. 配置路径
    OUTPUT_DIR = os.path.join(EXPORT_DIR, "tokenizer")

    # 2. 检查源路径
    if not os.path.exists(MODEL_DIR):
        print(f"❌ 模型目录不存在: {MODEL_DIR}")
        return

    print(f"   源路径: {MODEL_DIR}")
    print(f"   目标路径: {OUTPUT_DIR}")

    # 3. 创建目标目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # [核心增强] 生成单一 tokenizer.json 以便被 tokenizers 库加载
    try:
        from transformers import AutoTokenizer
        print("   正在将官方 Tokenizer 转换为单一格式 (tokenizer.json)...")
        # 直接从源目录加载官方分词器
        hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        # 强制保存，这会根据 vocab.json 和 merges.txt 自动合成 tokenizer.json
        hf_tokenizer.save_pretrained(OUTPUT_DIR)
        print("   ✅ 已合成单体 tokenizer.json。")
    except ImportError:
        print("   ⚠️  未安装 transformers，无法合成单体 tokenizer.json。")
    except Exception as e:
        print(f"   ⚠️  Tokenizer 转换失败: {e}")

    # 4. 需要复制的文本 Tokenizer 核心文件
    required_files = [
        "vocab.json",              # 词汇表
        "merges.txt",              # BPE 合并规则
        "tokenizer_config.json",   # Tokenizer 配置
        "config.json",             # 模型配置（包含 tokenizer 相关配置）
        "special_tokens_map.json", # 特殊 token 映射（如果存在）
    ]

    # 5. 复制文件
    copied_files = []
    skipped_files = []

    for file in required_files:
        src_path = os.path.join(MODEL_DIR, file)
        dst_path = os.path.join(OUTPUT_DIR, file)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            file_size = os.path.getsize(dst_path) / 1024 / 1024
            copied_files.append(file)
            print(f"   ✅ {file} ({file_size:.2f} MB)")
        else:
            skipped_files.append(file)
            print(f"   ⚠️  跳过 (不存在): {file}")

    # 6. 复制可选文件
    optional_files = [
        "generation_config.json",    # 生成配置
        "preprocessor_config.json", # 预处理配置
        "tokenizer.json",            # GPT-2 style tokenizer（如果存在）
    ]

    for file in optional_files:
        src_path = os.path.join(MODEL_DIR, file)
        dst_path = os.path.join(OUTPUT_DIR, file)

        if os.path.exists(src_path) and file not in copied_files:
            shutil.copy2(src_path, dst_path)
            file_size = os.path.getsize(dst_path) / 1024 / 1024
            copied_files.append(file)
            print(f"   ✅ {file} ({file_size:.2f} MB) [可选]")

    # 7. 总结
    print("\n" + "="*50)
    print(f"📊 复制完成统计:")
    print(f"   成功复制: {len(copied_files)} 个文件")
    print(f"   跳过文件: {len(skipped_files)} 个文件")
    print(f"   目标目录: {OUTPUT_DIR}")

    # 计算总大小
    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, f))
    ) / 1024 / 1024

    print(f"   总大小: {total_size:.2f} MB")
    print("="*50)

    if len(copied_files) >= 3:  # 至少有核心文件
        print("✅ 文本 Tokenizer 复制成功！")
    else:
        print("⚠️  警告：复制的文件较少，请检查源目录结构")

if __name__ == "__main__":
    main()
