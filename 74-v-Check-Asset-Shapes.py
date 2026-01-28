import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

def check_shapes():
    assets = {
        "master_head": "codec_head_weight.npy",
        "emb_0": "codec_embedding_0.npy",
        "proj": "craftsman_projection.npy" # 如果是大表拼接之前的投影层
    }
    
    print("--- Asset Shape Check ---")
    for name, filename in assets.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                print(f"{name} ({filename}): {data.shape}")
            except Exception as e:
                print(f"{name} ({filename}): Error loading - {e}")
        else:
            print(f"{name} ({filename}): MISSING")
            
    # 检查投影层 pt 文件 (来自 60/69 提取脚本)
    import torch
    proj_files = {
        "proj_w": "craftsman_hf/small_to_mtp_projection.weight.pt",
        "proj_b": "craftsman_hf/small_to_mtp_projection.bias.pt"
    }
    for name, rel_path in proj_files.items():
        path = os.path.join(PROJECT_ROOT, "model", rel_path)
        if os.path.exists(path):
            w = torch.load(path, map_location="cpu")
            print(f"{name} ({rel_path}): {w.shape}")
        else:
            print(f"{name} ({rel_path}): MISSING")

    # 检查所有 16 个分码表
    print("\n--- Codec Embedding Tables Check ---")
    for i in range(16):
        path = os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            print(f"Table {i}: {data.shape}")
        else:
            print(f"Table {i}: MISSING")

if __name__ == "__main__":
    check_shapes()
