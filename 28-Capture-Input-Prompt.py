import os
import sys
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

# 确保导入本地源码
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR)

# 捕获配置
SAVE_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")
os.makedirs(SAVE_DIR, exist_ok=True)

class Catcher:
    def __init__(self):
        self.inputs_embeds = None
        self.input_ids = None
        self.captured = False

catcher = Catcher()

def talker_forward_pre_hook(module, args, kwargs):
    if catcher.captured:
        return
        
    # Qwen3TTSTalkerModel.forward(self, input_ids=None, ..., inputs_embeds=None, ...)
    # Retrieve args
    input_ids = kwargs.get('input_ids', None)
    inputs_embeds = kwargs.get('inputs_embeds', None)
    
    # Handle positional args if kwargs are empty (rare but possible in some calls)
    if inputs_embeds is None and len(args) > 2:
         # forward signature: (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, ...)
         # Check modeling_qwen3_tts.py for exact sig if needed, but usually kwargs are used by HF
         pass

    # We want the Prefill step (Sequence Length > 1)
    if inputs_embeds is not None:
        seq_len = inputs_embeds.shape[1]
        if seq_len > 1:
            print(f"Captured Prefill: Shape {inputs_embeds.shape}")
            catcher.inputs_embeds = inputs_embeds.detach().to(torch.float32).cpu().numpy()
            
            if input_ids is not None:
                catcher.input_ids = input_ids.detach().cpu().numpy()
                print(f"Captured Input IDs: Shape {input_ids.shape}")
            
            catcher.captured = True

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-CustomVoice")
    
    print(f"载入官方模型进行 Prompt 捕获 (设备: {device})...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    
    try:
        tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=device, dtype=dtype)
        talker = tts.model.talker
        
        # Register Hook
        hooks = []
        hooks.append(talker.register_forward_pre_hook(talker_forward_pre_hook, with_kwargs=True))
        
        # Run Generation
        print("开始推理「今天天气不错」...")
        with torch.no_grad():
            tts.generate_custom_voice(
                text="今天天气不错",
                speaker="Vivian",
                language="Chinese",
                do_sample=False
            )
            
        if catcher.captured:
            # Save
            np.save(os.path.join(SAVE_DIR, "prompt_inputs_embeds.npy"), catcher.inputs_embeds)
            print(f"✅ Embeddings Captured: {catcher.inputs_embeds.shape}")
            
            if catcher.input_ids is not None:
                np.save(os.path.join(SAVE_DIR, "prompt_input_ids.npy"), catcher.input_ids)
                print(f"✅ Input IDs Captured: {catcher.input_ids.shape}")
                print(f"IDs: {catcher.input_ids[0]}")
        else:
            print("❌ Failed to capture prefill embeddings.")

    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
