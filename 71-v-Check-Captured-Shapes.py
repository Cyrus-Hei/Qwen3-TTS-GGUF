import os
import numpy as np

def check_captured_shapes():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
    
    print(f"--- 捕获数据维度自检 ---")
    
    # 检查每一步的关键数据
    for step in range(15):
        print(f"\n[Step {step}]")
        
        # 1. 投影层输入 (2048维)
        in_2048_path = os.path.join(SAVE_DIR, f"step_{step}_input_2048.npy")
        if os.path.exists(in_2048_path):
            data = np.load(in_2048_path)
            print(f"  Input 2048 Shape: {data.shape}")
        
        # 2. Transformer 输出 (1024维)
        out_hidden_path = os.path.join(SAVE_DIR, f"step_{step}_output_hidden.npy")
        if os.path.exists(out_hidden_path):
            data = np.load(out_hidden_path)
            print(f"  Output Hidden Shape: {data.shape}")
            
        # 3. 最终生成的 Token ID
        out_ids_path = os.path.join(SAVE_DIR, f"step_{step}_output_ids.npy")
        if os.path.exists(out_ids_path):
            data = np.load(out_ids_path)
            print(f"  Output Token IDs: {data} (Shape: {data.shape})")

if __name__ == "__main__":
    check_captured_shapes()
