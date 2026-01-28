import os
import sys
import torch
import numpy as np
from pathlib import Path

# 设置搜索路径
PROJECT_ROOT = Path(__file__).parent
SOURCE_DIR = PROJECT_ROOT / "Qwen3-TTS"
sys.path.append(str(SOURCE_DIR))

# 导入核心模型
try:
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
except ImportError:
    print(f"❌ 导入失败。请检查 {SOURCE_DIR} 路径。")
    sys.exit(1)

# 配置
MODEL_PATH = PROJECT_ROOT / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = PROJECT_ROOT / "model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_master_head():
    print(f"🚀 正在加载官方模型 (CPU) 以提取 Master Head: {MODEL_PATH}")
    
    try:
        # 仅加载权重，不需要 full wrapper 
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # 提取关键权重：Master LM Head (决定 Code 0 的预测概率)
        # 路径: model.talker.codec_head
        master_head_weight = model.talker.codec_head.weight
        
        print(f"📊 Head 权重形状: {master_head_weight.shape}")
        
        # 保存为 codec_head.npy
        save_path = OUTPUT_DIR / "codec_head.npy"
        np.save(save_path, master_head_weight.detach().numpy())
        
        print(f"✅ 成功导出 Master Head 至: {save_path}")
        
        # 同时校验与 Embedding 0 是否绑定 (Tied)
        emb_0_weight = model.talker.get_input_embeddings().weight
        is_tied = torch.allclose(master_head_weight, emb_0_weight, atol=1e-8)
        
        if is_tied:
            print("💡 注意: 该模型权重已绑定 (Tied)，Head 与 Embedding 0 数值一致。")
        else:
            print("💡 结论: 该模型权重未绑定 (Not Tied)，必须使用专用的 codec_head.npy 才能正确推理。")

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_master_head()
