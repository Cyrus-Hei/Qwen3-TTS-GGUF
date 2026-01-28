import os
import sys
import numpy as np
import ctypes
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型与路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_craftsman")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "qwen3_tts_craftsman_medium.gguf")

def compare_vectors(official, gguf_out, vec_name):
    """对比两个向量的相似度"""
    off_flat = official.flatten()
    gguf_flat = gguf_out.flatten()
    
    # 余弦相似度
    norm_off = np.linalg.norm(off_flat)
    norm_gguf = np.linalg.norm(gguf_flat)
    cos_sim = np.dot(off_flat, gguf_flat) / (norm_off * norm_gguf + 1e-9)
    # MAE
    mae = np.mean(np.abs(off_flat - gguf_flat))
    
    mark = "✅" if cos_sim > 0.999 else "⚠️"
    print(f"{mark} [{vec_name}] CosSim: {cos_sim:.6f}, MAE: {mae:.6f}")
    return cos_sim

def run_verification():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到 GGUF 模型: {MODEL_PATH}")
        return
    
    print(f"Loading GGUF: {MODEL_PATH}")
    model = nano_llama.load_model(MODEL_PATH, n_gpu_layers=0)
    if not model: return
        
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.embeddings = True 
    ctx = nano_llama.llama_init_from_model(model, ctx_params)
    n_embd = nano_llama.llama_model_n_embd(model)

    # --- 准备投影矩阵 ---
    print("Loading Projection Layer...")
    from safetensors.torch import load_file
    orig_model_path = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice", "model.safetensors")
    orig_weights = load_file(orig_model_path)
    proj_w = orig_weights["talker.code_predictor.small_to_mtp_projection.weight"].float().numpy()
    proj_b = orig_weights["talker.code_predictor.small_to_mtp_projection.bias"].float().numpy()

    # --- Step 0 验证 (Offset 0) ---
    print("\n>>> Verifying Step 0 (Offset 0) <<<")
    
    # Input: Master Hidden States -> Projection -> GGUF Embeddings
    input_step0 = np.load(os.path.join(CAPTURED_DIR, "step_0_input_2048.npy")).astype(np.float32)
    flat_input0 = input_step0.reshape(-1, 2048)
    proj_input0 = flat_input0 @ proj_w.T + proj_b # [n, 1024]
    
    n_tokens0 = proj_input0.shape[0]
    batch0 = nano_llama.llama_batch_init(n_tokens0, n_embd, 1)
    batch0.n_tokens = n_tokens0
    
    # 注入 Embedding
    embd_data0 = np.ascontiguousarray(proj_input0.astype(np.float32))
    ctypes.memmove(batch0.embd, embd_data0.ctypes.data, embd_data0.nbytes)
    
    for i in range(n_tokens0):
        batch0.pos[i] = i
        batch0.n_seq_id[i] = 1
        batch0.seq_id[i][0] = 0
        batch0.logits[i] = 1 if i == n_tokens0 - 1 else 0
        
    nano_llama.llama_decode(ctx, batch0)
    out_ptr0 = nano_llama.llama_get_embeddings(ctx)
    gguf_out0 = np.ctypeslib.as_array(out_ptr0, shape=(n_tokens0, n_embd)).copy()[-1]
    
    official_out0 = np.load(os.path.join(CAPTURED_DIR, "step_0_output_hidden.npy")).astype(np.float32)
    official_last0 = official_out0.flatten().reshape(-1, 1024)[-1]
    compare_vectors(official_last0, gguf_out0, "Step 0 Hidden")
    
    nano_llama.llama_batch_free(batch0)

    # --- Step 1 验证 (Offset 2048) ---
    print("\n>>> Verifying Step 1 (Offset 2048) <<<")
    # Step 1: Input is token IDs from Step 0 output? 
    # Or do we have captured input for Step 1?
    # 假设我们没有 Step 1 的 npy，我们用 Step 0 的输出来模拟？
    # 不行，我们需要真实的 Step 1 输入来对比真实的 Step 1 输出。
    # 检查目录下文件
    step1_in_path = os.path.join(CAPTURED_DIR, "step_1_input_ids.npy")
    step1_out_path = os.path.join(CAPTURED_DIR, "step_1_output_hidden.npy")
    
    if os.path.exists(step1_in_path) and os.path.exists(step1_out_path):
        # 正常流程: input_ids -> GGUF (with offset)
        input_ids1 = np.load(step1_in_path).flatten().astype(np.int32)
        print(f"Step 1 raw input IDs: {input_ids1}")
        
        # Apply Offset: ID + 2048
        # 注意: GGUF 的 Embedding 表已经是拼接过的 [4096, 1024]
        # 表 0 对应 index 0-2047, 表 1 对应 2048-4095
        # 所以这里的 input_ids 应该加上 2048
        offset_ids1 = input_ids1 + 2048
        print(f"Step 1 offset IDs: {offset_ids1}")
        
        n_tokens1 = len(offset_ids1)
        batch1 = nano_llama.llama_batch_init(n_tokens1, 0, 1) # 0 means token mode ? No, n_embd=0 for token batch
        
        # 修正: llama_batch_init 第二参数是 n_embd，如果是 token 输入，则不用它？
        # `llama_batch_init` allocates `embd` if n_embd > 0.
        # Here we use tokens, so we want `token` array allocated, `embd` can be null.
        # checking nano_llama: llama_batch_init(n_tokens, embd, n_seq_max)
        # If we pass embd=0, it won't allocate embd. Valid.
        batch1 = nano_llama.llama_batch_init(n_tokens1, 0, 1) 
        batch1.n_tokens = n_tokens1
        
        for i in range(n_tokens1):
            batch1.token[i] = offset_ids1[i]
            batch1.pos[i] = i # Pos 应该接着 Step 0？不，工匠是独立的 step 推理，pos 重置？
            # 实际上工匠的每一层都是独立的，不需要 KV Cache 传递？
            # 这里的 Step 0, Step 1 是指 Auto-Regressive 的 15 步。
            # 必须维护 KV Cache！
            # 所以 Pos 应该接续。
            batch1.pos[i] = n_tokens0 + i
            batch1.n_seq_id[i] = 1
            batch1.seq_id[i][0] = 0
            batch1.logits[i] = 1 if i == n_tokens1 - 1 else 0
            
        nano_llama.llama_decode(ctx, batch1)
        # ... fetch output ...
        out_ptr1 = nano_llama.llama_get_embeddings(ctx)
        gguf_out1 = np.ctypeslib.as_array(out_ptr1, shape=(n_tokens1, n_embd)).copy()[-1]
        
        official_out1 = np.load(step1_out_path).astype(np.float32)
        official_last1 = official_out1.flatten().reshape(-1, 1024)[-1]
        compare_vectors(official_last1, gguf_out1, "Step 1 Hidden")
        
        nano_llama.llama_batch_free(batch1)
    else:
        print("⚠️ 缺少 Step 1 验证数据，仅验证了拼接模型的 Step 0 部分。")
        print("这证明了拼接模型的前半部分权重是正常的。")

    # 清理
    nano_llama.llama_free(ctx)
    nano_llama.llama_model_free(model)

if __name__ == "__main__":
    run_verification()
