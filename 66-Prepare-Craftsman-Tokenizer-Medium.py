import os
import json

def create_craftsman_tokenizer_medium():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    TARGET_DIR = os.path.join(PROJECT_ROOT, "model", "craftsman_medium_hf")
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    print(f"--- 正在构造中级工匠迷你分词器 (目标覆盖 4096 范围) ---")
    
    # 依然生成极少量 Token，依赖 convert_hf_to_gguf 的自动 padding
    # 但为了稳妥，我们在 tokenizer_config 里不显式声明 vocab_size (让它自己算或者用 default)
    # 或者我们声明一下，但不影响 tokenizer.json
    
    vocab = {}
    # 添加 BPE 基础字符和合并后的字符，确保 merge 规则有效
    vocab["a"] = 0
    vocab["b"] = 1
    vocab["ab"] = 2
    
    # 填充剩下的以满足少量 token 需求
    for i in range(3, 15):
        vocab[f"<t_{i}>"] = i
        
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": ["a b"] # 合法的 BPE 规则
        }
    }
    
    with open(os.path.join(TARGET_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    
    with open(os.path.join(TARGET_DIR, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "Qwen2Tokenizer"}, f)
        
    print(f"✅ 中级工匠假分词器构造完成: {TARGET_DIR}")

if __name__ == "__main__":
    create_craftsman_tokenizer_medium()
