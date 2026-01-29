"""
81-Verify-Stateful-PyTorch.py
验证 StatefulCodecONNXWrapper 的流式推理正确性（参考87号脚本用法）。
"""
import os
import torch
import numpy as np
import soundfile as sf
from qwen3_tts_gguf.codec_export import StatefulCodecONNXWrapper
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

    # 3. 加载模型
    print(f"🚀 正在加载模型: {MODEL_DIR}...")
    tokenizer_model_dir = os.path.join(MODEL_DIR, "speech_tokenizer")
    load_path = tokenizer_model_dir if os.path.exists(tokenizer_model_dir) else MODEL_DIR
    model = Qwen3TTSTokenizerV2Model.from_pretrained(load_path)

    # 初始化新版 Stateful ONNX Wrapper
    wrapper = StatefulCodecONNXWrapper(model).eval()
    device = next(wrapper.parameters()).device

    # 获取配置参数
    num_layers = wrapper.num_layers
    cfg = wrapper.decoder.config
    num_heads = cfg.num_key_value_heads if hasattr(cfg, 'num_key_value_heads') else cfg.num_attention_heads
    head_dim = cfg.head_dim

    print(f"   模型配置: num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}")

    # 4. 模拟流式推理
    print("🧪 正在执行流式分波段推理对比...")

    chunk_size = 3
    all_codes_len = len(codes_np)

    # 初始化状态（参考87号脚本的初始化方式）
    def init_states():
        pre_conv_h = torch.zeros(1, 512, 0, device=device)
        latent_buf = torch.zeros(1, 1024, 0, device=device)
        conv_h = torch.zeros(1, 1024, 0, device=device)
        pkv = []
        for _ in range(num_layers):
            pkv.append(torch.zeros(1, num_heads, 0, head_dim, device=device))  # K
        for _ in range(num_layers):
            pkv.append(torch.zeros(1, num_heads, 0, head_dim, device=device))  # V
        return pre_conv_h, latent_buf, conv_h, pkv

    pre_conv_h, latent_buf, conv_h, pkv_list = init_states()
    all_chunks_audio = []

    with torch.no_grad():
        for i in range(0, all_codes_len, chunk_size):
            end = min(i + chunk_size, all_codes_len)
            is_last = (end == all_codes_len)

            # 准备当前 chunk 的输入
            chunk_codes = torch.from_numpy(codes_np[i:end]).unsqueeze(0).to(device).long()
            is_last_tensor = torch.tensor([1.0 if is_last else 0.0], device=device)

            print(f"  > 处理分步: 帧 {i} 到 {end-1} (is_last={is_last})")

            # 执行有状态推理（新版 API：展开的 KV Cache 参数）
            outputs = wrapper(
                chunk_codes,
                is_last_tensor,
                pre_conv_h,
                latent_buf,
                conv_h,
                *pkv_list
            )

            # 解包输出（参考87号脚本的解包方式）
            chunk_wav_full = outputs[0]
            valid_samples = int(outputs[1].item())
            pre_conv_h = outputs[2]
            latent_buf = outputs[3]
            conv_h = outputs[4]
            pkv_list = outputs[5:]

            # 提取有效音频
            chunk_wav = chunk_wav_full[:, :valid_samples].cpu().numpy().squeeze()

            if valid_samples > 0:
                all_chunks_audio.append(chunk_wav)
                print(f"      └─ 输出 {chunk_wav.size} 采样点")
            else:
                print(f"      └─ 累积中，暂无输出")

    # 5. 拼接与对比
    full_audio_stream = np.concatenate(all_chunks_audio)

    print("\n" + "="*40)
    print(f"📊 流式拼接总长度: {len(full_audio_stream)}")
    print(f"📊 参考波形总长度: {len(ref_wav_np)}")

    # 计算差异
    common_len = min(len(full_audio_stream), len(ref_wav_np))
    diff = np.abs(full_audio_stream[:common_len] - ref_wav_np[:common_len])
    mse = np.mean(diff**2)
    max_diff = np.max(diff)

    print(f"✅ 验证结果 (基于共同长度 {common_len}):")
    print(f"   - MSE: {mse:.2e}")
    print(f"   - Max Diff: {max_diff:.2e}")

    # 寻找第一个显著差异的位置 (阈值 1e-4)
    threshold = 1e-4
    divergence_idx = np.where(diff > threshold)[0]
    if len(divergence_idx) > 0:
        first_idx = divergence_idx[0]
        print(f"\n⚠️ 首次出现显著差异的位置: {first_idx}")
        print(f"   - 时间点: {first_idx / 24000:.4f} 秒")
        print(f"   - 差异幅度: {diff[first_idx]:.6f}")
        # 估算对应的 code 帧 (假设每帧对应 N 个采样点)
        samples_per_frame = len(ref_wav_np) / len(codes_np)
        print(f"   - 对应代码帧索引: 约第 {first_idx / samples_per_frame:.2f} 帧")

    print("="*40)

    if max_diff < 1e-4:
        print("\n🎉 成功！流式状态机逻辑验证通过。")
    else:
        print("\n⚠️ 警报：流式输出与参考输出在重合区域不一致。")

    # 保存结果供人工听感检查
    output_dir = "output_verify"
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, "verify_stateful_stream.wav"), full_audio_stream, 24000)
    print(f"\n检查音频已保存至: {output_dir}")

if __name__ == "__main__":
    main()
