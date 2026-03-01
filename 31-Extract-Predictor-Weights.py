import os
import json
import torch
import numpy as np
from safetensors.torch import save_file, load_file
from export_config import MODEL_DIR, EXPORT_DIR

def extract_predictor_hf():
    MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
    OUTPUT_DIR = os.path.join(EXPORT_DIR, "predictor_hf")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(EXPORT_DIR, 'embeddings'), exist_ok=True)

    print(f"--- 正在提取 Predictor 权重 (兼容模式) ---")
    
    weights = load_file(MODEL_PATH)
    
    # 1. 探测投影层 (1.7B 有, 0.6B 没有)
    proj_key = "talker.code_predictor.small_to_mtp_projection.weight"
    has_projection = proj_key in weights
    
    if has_projection:
        print("💡 检测到投影层 (1.7B 模式)，开启预投影计算...")
        proj_w = weights[proj_key]
        proj_b = weights["talker.code_predictor.small_to_mtp_projection.bias"]
        # 导出 Numpy 资产供推理使用
        np.save(os.path.join(EXPORT_DIR, 'embeddings', "proj_weight.npy"), proj_w.float().numpy())
        np.save(os.path.join(EXPORT_DIR, 'embeddings', "proj_bias.npy"), proj_b.float().numpy())
    else:
        print("💡 未发现投影层 (0.6B 模式)，跳过预投影...")

    # 2. 动态统计词表组数 (Codec Embedding Groups)
    emb_keys = [k for k in weights.keys() if "code_predictor.model.codec_embedding." in k and ".weight" in k]
    num_emb_groups = len(emb_keys)
    print(f"   检测到 Codec 组数: {num_emb_groups}")

    # 3. 处理 Embedding 和 LM Head
    emb_list = []
    head_list = []
    
    for i in range(num_emb_groups):
        # 处理 Embedding
        emb_key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
        raw_emb = weights[emb_key]
        
        if has_projection:
            # 1.7B: 执行预投影
            target_emb = torch.nn.functional.linear(raw_emb, proj_w, proj_b)
        else:
            # 0.6B: 直接使用
            target_emb = raw_emb
        
        emb_list.append(target_emb)
        
        # 处理 LM Head
        head_key = f"talker.code_predictor.lm_head.{i}.weight"
        head_list.append(weights[head_key])
        
    cat_emb = torch.cat(emb_list, dim=0)
    cat_head = torch.cat(head_list, dim=0)
    
    print(f"   拼接后的 Embedding 形状: {cat_emb.shape}")
    print(f"   拼接后的 LM Head 形状: {cat_head.shape}")

    # 4. 统计层数
    layer_indices = set()
    for k in weights.keys():
        if "code_predictor.model.layers." in k:
            parts = k.split('.')
            # talker.code_predictor.model.layers.N...
            idx = int(parts[4])
            layer_indices.add(idx)
    num_layers = len(layer_indices)
    hidden_size = cat_emb.shape[1]
    vocab_size = cat_emb.shape[0]
    print(f"   检测到层数: {num_layers}, 隐藏层维度: {hidden_size}")

    # 5. 组装新权重
    new_weights = {
        "embed_tokens.weight": cat_emb,
        "lm_head.weight": cat_head,
        "norm.weight": weights["talker.code_predictor.model.norm.weight"]
    }
    
    # 搬运 Backbone Layers
    for i in range(num_layers):
        prefix = f"talker.code_predictor.model.layers.{i}."
        for k, v in weights.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, f"layers.{i}.")
                new_weights[new_key] = v

    save_file(new_weights, os.path.join(OUTPUT_DIR, "model.safetensors"))
    
    # 6. 构造配置
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 3, # Qwen3 通常是 3 倍
        "num_hidden_layers": num_layers,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": hidden_size // 16,
        "rms_norm_eps": 1e-06,
        "vocab_size": vocab_size,
        "rope_theta": 1000000.0,
        "use_cache": True,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "max_position_embeddings": 32768
    }
    
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"✅ Predictor 提取完成: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_predictor_hf()
