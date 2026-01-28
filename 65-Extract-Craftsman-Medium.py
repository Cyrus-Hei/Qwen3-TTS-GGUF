import os
import json
import torch
from safetensors.torch import save_file, load_file

def extract_craftsman_hf_medium():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "model.safetensors")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model", "craftsman_medium_hf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- 正在提取中级工匠权重 (2组表拼接) ---")
    
    weights = load_file(MODEL_PATH)
    
    # 1. 准备投影层
    proj_w = weights["talker.code_predictor.small_to_mtp_projection.weight"] # [1024, 2048]
    proj_b = weights["talker.code_predictor.small_to_mtp_projection.bias"]   # [1024]
    
    # 2. 处理 Embedding (拼接 0 和 1)
    # 目标: [4096, 1024]
    emb_list = []
    for i in range(2):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        raw_emb = weights[key] # [2048, 2048]
        print(f"处理 Embedding {i}: 预投影...")
        # Y = X @ W.T + b
        proj_emb = torch.nn.functional.linear(raw_emb, proj_w, proj_b) # [2048, 1024]
        emb_list.append(proj_emb)
        
    cat_emb = torch.cat(emb_list, dim=0) # [4096, 1024]
    print(f"拼接后的 Embedding 形状: {cat_emb.shape}")
    
    # 3. 处理 LM Head (拼接 0 和 1)
    # 目标: [4096, 1024]
    head_list = []
    for i in range(2):
        key = f"talker.code_predictor.lm_head.{i}.weight"
        raw_head = weights[key] # [2048, 1024]
        head_list.append(raw_head)
    
    cat_head = torch.cat(head_list, dim=0)
    print(f"拼接后的 LM Head 形状: {cat_head.shape}")
    
    # 4. 组装权重字典
    new_weights = {
        "embed_tokens.weight": cat_emb,
        "lm_head.weight": cat_head,
        "norm.weight": weights["talker.code_predictor.model.norm.weight"]
    }
    
    # 提取 Backbone (不变)
    for i in range(5):
        prefix = f"talker.code_predictor.model.layers.{i}."
        for k, v in weights.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, f"layers.{i}.")
                new_weights[new_key] = v

    # 保存
    save_file(new_weights, os.path.join(OUTPUT_DIR, "model.safetensors"))
    
    # 5. 构造 Config
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 5,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "vocab_size": 4096, # 关键: 2048 * 2
        "rope_theta": 1000000.0,
        "use_cache": True,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "max_position_embeddings": 32768
    }
    
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"✅ 中级工匠 HF 格式提取完成: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_craftsman_hf_medium()
