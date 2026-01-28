import os
import ctypes
import numpy as np
import torch
import qwen3_tts_gguf.nano_llama as nano_llama

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MASTER_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_talker.gguf")
CRAFTSMAN_GGUF = os.path.join(MODEL_DIR, "qwen3_tts_craftsman_advanced.gguf")
MASTER_HEAD_PATH = os.path.join(MODEL_DIR, "codec_head_weight.npy")
PROJ_PT_PATH = os.path.join(MODEL_DIR, "craftsman_hf/master_to_craftsman_proj.pt")

def load_assets():
    print("正在加载外部权重资产...")
    assets = {
        "master_head": np.load(MASTER_HEAD_PATH),
        "emb_tables": [np.load(os.path.join(MODEL_DIR, f"codec_embedding_{i}.npy")) for i in range(16)],
        "proj": torch.load(PROJ_PT_PATH, map_location="cpu")
    }
    print("✅ 资产加载完成。")
    return assets

def apply_projection(hidden_2048, proj_assets):
    """将 2048 维隐藏层投影到 1024 维"""
    w = proj_assets["weight"].float().numpy() # [1024, 2048]
    b = proj_assets["bias"].float().numpy()   # [1024]
    return hidden_2048 @ w.T + b

def run_joint_inference():
    print("=== [74] 大师与工匠联合推理闭环测试 ===\n")
    
    # 1. 加载模型与资产
    assets = load_assets()
    
    master_model = nano_llama.load_model(MASTER_GGUF, n_gpu_layers=0)
    craftsman_model = nano_llama.load_model(CRAFTSMAN_GGUF, n_gpu_layers=0)
    
    ctx_params = nano_llama.llama_context_default_params()
    ctx_params.n_ctx = 1024
    ctx_params.embeddings = True
    
    m_ctx = nano_llama.llama_init_from_model(master_model, ctx_params)
    c_ctx = nano_llama.llama_init_from_model(craftsman_model, ctx_params)
    
    m_embd_dim = 2048
    c_embd_dim = 1024
    
    # --- 步骤 A: 大师 Prefill & 产生第一个 Code ---
    print("\n--- 步骤 A: 大师推理 ---")
    m_batch = nano_llama.llama_batch_init(1, m_embd_dim, 1)
    m_batch.n_tokens = 1
    # 构造随机起始输入 (模拟 Prompt 推理后的最后一帧)
    start_hidden = np.random.randn(1, m_embd_dim).astype(np.float32)
    ctypes.memmove(m_batch.embd, start_hidden.ctypes.data, start_hidden.nbytes)
    m_batch.pos[0] = 0
    m_batch.n_seq_id[0] = 1
    m_batch.seq_id[0][0] = 0
    m_batch.logits[0] = 1
    
    nano_llama.llama_decode(m_ctx, m_batch)
    m_out_ptr = nano_llama.llama_get_embeddings(m_ctx)
    m_hidden = np.ctypeslib.as_array(m_out_ptr, shape=(m_embd_dim,)).copy()
    
    # 预测第一个 Code (Table 0)
    logits_0 = m_hidden @ assets["master_head"].T
    code_0 = np.argmax(logits_0)
    print(f"大师产出的 Code 0: {code_0}")
    
    # 查原始 2048 词表
    emb_0 = assets["emb_tables"][0][code_0] # [2048]
    
    # --- 步骤 B: 构造工匠输入并投影 ---
    print("\n--- 步骤 B: 工匠推理 (15 步自回归) ---")
    # 工匠输入拼接: [Master Hidden, Emb 0] -> [2, 2048]
    combined_input = np.stack([m_hidden, emb_0], axis=0) # [2, 2048]
    # 执行维度投影 -> [2, 1024]
    projected_input = apply_projection(combined_input, assets["proj"]) # [2, 1024]
    
    all_codes = [code_0]
    
    # 给工匠起跑
    c_batch = nano_llama.llama_batch_init(16, c_embd_dim, 1)
    # Step 0: Prefill 2 tokens
    c_batch.n_tokens = 2
    ctypes.memmove(c_batch.embd, projected_input.ctypes.data, projected_input.nbytes)
    for i in range(2):
        c_batch.pos[i] = i
        c_batch.n_seq_id[i] = 1
        c_batch.seq_id[i][0] = 0
        c_batch.logits[i] = 1 if i == 1 else 0
        
    nano_llama.llama_decode(c_ctx, c_batch)
    
    # 获取第一个预测 (Table 1)
    c_logits_ptr = nano_llama.llama_get_logits(c_ctx)
    # 工匠总词表 30720, 每段 2048
    all_logits = np.ctypeslib.as_array(c_logits_ptr, shape=(2, 30720)) 
    last_logits = all_logits[1] # 取最后一个 token
    
    # 后续 14 步自回归
    for step in range(1, 15):
        # 提取当前 Table 的分布
        table_logits = last_logits[step * 2048 : (step + 1) * 2048]
        code = np.argmax(table_logits)
        all_codes.append(code)
        
        # 构造下一步输入
        # GGUF 内部自回归: 需要用当前 code 在 i*2048 处的 Embedding
        # 注意：这里我们是在验证 GGUF 内部流程，为了简化，我们直接喂给 llama_decode
        c_batch.n_tokens = 1
        c_batch.pos[0] = step + 1
        # 获取 GGUF 的 Embedding (1024维)：这里我们需要之前验证过的偏移
        # 或者从外部获取 proj 后的 Embedding 再次喂入
        # 为了演示，我们从 GGUF 推理出的 ID 对应的 Embedding 获取 (这里简化为拉取 GGUF 权重)
        # 实际推理中，llama.cpp 会自动处理 input_ids，但我们现在用 embd 模式。
        # 这里手动从原始 2048 查表再投影是比较保险的：
        next_emb_2048 = assets["emb_tables"][step][code]
        next_emb_1024 = apply_projection(next_emb_2048, assets["proj"])
        
        ctypes.memmove(c_batch.embd, next_emb_1024.ctypes.data, next_emb_1024.nbytes)
        c_batch.logits[0] = 1
        
        nano_llama.llama_decode(c_ctx, c_batch)
        last_logits = np.ctypeslib.as_array(nano_llama.llama_get_logits(c_ctx), shape=(30720,))

    # 获取最后一步产出的 code
    final_code = np.argmax(last_logits[14 * 2048 : 15 * 2048])
    all_codes.append(final_code)
    
    print(f"生成的完整分码列表 (16个): {all_codes}")

    # --- 步骤 C: 求和反馈给大师 ---
    print("\n--- 步骤 C: 反馈求和并驱动大师 ---")
    summed_vector = np.zeros(m_embd_dim, dtype=np.float32)
    for i, code in enumerate(all_codes):
        summed_vector += assets["emb_tables"][i][code]
    
    # 此处省略 trailing_text 补偿，直接验证连通性
    m_batch.n_tokens = 1
    m_batch.pos[0] = 1 # KV Cache 位置推进
    ctypes.memmove(m_batch.embd, summed_vector.ctypes.data, summed_vector.nbytes)
    m_batch.logits[0] = 1
    
    nano_llama.llama_decode(m_ctx, m_batch)
    print("✅ 大师成功接收求和向量并完成二次推理！")

    # 清理
    nano_llama.llama_free(m_ctx)
    nano_llama.llama_free(c_ctx)
    nano_llama.llama_model_free(master_model)
    nano_llama.llama_model_free(craftsman_model)

if __name__ == "__main__":
    run_joint_inference()
