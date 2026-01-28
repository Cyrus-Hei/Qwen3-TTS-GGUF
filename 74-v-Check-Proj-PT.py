import torch
import os

path = "model/craftsman_hf/master_to_craftsman_proj.pt"
if os.path.exists(path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{k}: {v.shape}")
    elif torch.is_tensor(data):
        print(f"Tensor shape: {data.shape}")
    else:
        print(f"Unknown type: {type(data)}")
else:
    print("File not found.")
