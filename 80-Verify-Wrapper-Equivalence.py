import os
import torch
import numpy as np
import soundfile as sf
from qwen3_tts_gguf.codec_export import CodecExportWrapper, StreamingCodecExportWrapper
from qwen3_tts_gguf.tokenizer_12hz.modeling_tokenizer import Qwen3TTSTokenizerV2Model

def main():
    # 1. 配置路径
    MODEL_DIR = r'./Qwen3-TTS-12Hz-1.7B-CustomVoice'
    DEBUG_DATA_DIR = "debug_data"
    
    codes_path = os.path.join(DEBUG_DATA_DIR, "jintian_codes.npy")
    ref_wav_path = os.path.join(DEBUG_DATA_DIR, "jintian_ref.wav")
    
    if not os.path.exists(codes_path):
        print(f"❌ 找不到调试数据: {codes_path}，请先运行 43-Generate-Debug-Data.py")
        return

    # 2. 加载数据
    codes_np = np.load(codes_path)
    ref_wav_np, sr = sf.read(ref_wav_path)
    
    # [T, Q] -> [B=1, T, Q]
    codes_torch = torch.from_numpy(codes_np).unsqueeze(0).long()
    
    print(f"📊 输入 Shape: {codes_torch.shape}")
    print(f"📊 参考波形长度: {len(ref_wav_np)}")

    # 3. 加载模型
    print(f"🚀 正在加载模型: {MODEL_DIR}...")
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    
    model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)
    model.eval()
    
    # 4. 初始化 Wrapper
    wrapper_orig = CodecExportWrapper(model).eval()
    wrapper_stream = StreamingCodecExportWrapper(model).eval()
    
    # 5. 推理对比
    print("🧪 正在执行推理对比...")
    with torch.no_grad():
        # 原始 Wrapper 推理
        out_orig = wrapper_orig(codes_torch).numpy().squeeze()
        
        # 流式验证 Wrapper 推理 (内部有逐帧循环)
        out_stream = wrapper_stream(codes_torch).numpy().squeeze()
        
    # 6. 数值比对
    # 注意：即便逻辑完全一致，逐帧循环与全量计算在浮点数累加顺序上可能有极微小差异
    mse = np.mean((out_orig - out_stream)**2)
    max_diff = np.max(np.abs(out_orig - out_stream))
    
    # 与参考波形（之前的 ONNX 输出）比对
    mse_ref = np.mean((out_orig - ref_wav_np)**2)
    
    print("\n" + "="*40)
    print(f"✅ 验证结果:")
    print(f"   - [新旧 Wrapper 对比] MSE: {mse:.2e} | Max Diff: {max_diff:.2e}")
    print(f"   - [原始 Wrapper vs 参考音频] MSE: {mse_ref:.2e}")
    print("="*40)
    
    if max_diff < 1e-5:
        print("\n🎉 恭喜！新 Wrapper 的拆解逻辑与原版完全等价。")
        print("这证明了 Transformer 的 KV Cache 逐帧推理逻辑已经接通。")
    else:
        print("\n⚠️ 注意：数值存在微小差异，请确认是否在可接受范围内。")
        
    # 保存结果供人工听感检查
    output_dir = "output_verify"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, "verify_streaming_wrapper.wav"), out_stream, 24000)
    print(f"\n音频已保存至: {output_dir}")

if __name__ == "__main__":
    main()
