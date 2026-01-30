import os
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CAPTURED_DIR = os.path.join(PROJECT_ROOT, "captured_assembly")

def main():
    print("--- Input Prompt Verification ---")
    
    # 1. Load Data
    print("[1/3] Loading Tables...")
    try:
        text_table = np.load(os.path.join(MODEL_DIR, "text_embedding_projected.npy"))
        codec_table = np.load(os.path.join(MODEL_DIR, "codec_embedding_0.npy"))
        
        captured_path = os.path.join(CAPTURED_DIR, "prompt_inputs_embeds.npy")
        if not os.path.exists(captured_path):
            print("❌ Captured data not found. Run 28-Capture-Input-Prompt.py first.")
            return
            
        official_embeds = np.load(captured_path)
        print(f"  Official Embeds Shape: {official_embeds.shape}")
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return

    # 2. Construct Manual Embedding
    print("[2/3] Constructing Manual Block...")
    
    # Mapping from 02-Manual-Embedding-Inference.py
    # Text: "今天天气不错" (Vivian, Chinese) -- Wait, 02 was "今天天气好", let's check ids
    # The user asked for "今天天气不错" in 28-Capture.
    # Tokenizer IDs will differ for "不错" (bu cuo) vs "好" (hao).
    # I don't have the tokenizer running here to debug, so I have to guess or rely on captured input_ids if they exist
    
    # Let's check if input_ids were captured.
    # The previous run didn't print "✅ Input IDs Captured", so likely they were None in kwargs.
    # HF generate() usually passes input_ids associated with past_key_values, 
    # but for the first step (prefill), it passes input_ids.
    # Hook caught `inputs_embeds`. 
    
    # If I can't be sure of IDs, I can try to brute-force match or just check if the logic holds for the structure.
    # Actually, for "Verify", I should have used the *exact same* IDs.
    
    # BUT, the goal is to verify the *summation logic*.
    # I can verify if `official_embeds[0, i]` = `text_table[t] + codec_table[c]` for SOME t, c.
    # Since I have the tables, I can iterate and find best match for each position? 
    # That's too expensive (150k vocab).
    
    # Alternative: Use "Blind Substraction".
    # official[i] = text[t] + codec[c]
    # We know c for most positions is C_PAD (2148) or Special Tokens.
    # Let's assume the Codec IDs from 02 are structurally similar (Control tokens).
    
    # Structure:
    # 0: <|im_start|> (151644) + None
    # 1: assistant (77091) + None
    # 2: \n (198) + None
    # 3: Think (151671, 2154)
    # 4: Think_BOS (151671, 2156)
    # 5: Lang (151671, 2055) - Chinese
    # 6: Think_EOS (151671, 2157)
    # 7: Speaker (151671, 3065) - Vivian
    # 8: TTS_BOS (151672, 2148)
    # ... Text Content ...
    # N: TTS_EOS (151673, 2148)
    # N+1: Codec_BOS (151671, 2149)
    
    mapping_template = [
        ("Start", 151644, None),   # 0
        ("Role",  77091, None),    # 1
        ("NL",    198, None),      # 2
        ("Think", 151671, 2154),   # 3
        ("TBOS",  151671, 2156),   # 4
        ("Lang",  151671, 2055),   # 5 (Chinese)
        ("TEOS",  151671, 2157),   # 6
        ("Spk",   151671, 3065),   # 7 (Vivian)
        ("AudioB",151672, 2148),   # 8
        # Content...
    ]
    
    # We can verify these fixed positions first.
    
    match_count = 0
    total_positions = official_embeds.shape[1] # 14
    
    print(f"  Verifying Fixed Headers (0-8)...")
    for i, (name, t_id, c_id) in enumerate(mapping_template):
        if i >= total_positions: break
        
        target = official_embeds[0, i]
        
        t_vec = text_table[t_id] if t_id is not None else np.zeros(2048)
        c_vec = codec_table[c_id] if c_id is not None else np.zeros(2048)
        
        mine = t_vec + c_vec
        
        # Compare
        # Using Cosine Similarity and MSE
        dot = np.dot(target, mine)
        norm_t = np.linalg.norm(target)
        norm_m = np.linalg.norm(mine)
        sim = dot / (norm_t * norm_m + 1e-9)
        
        diff = np.abs(target - mine)
        mae = np.mean(diff)
        
        status = "✅" if sim > 0.9999 else "❌"
        print(f"    Pos {i} [{name}]: Sim={sim:.6f}, MAE={mae:.6f} {status}")
        if sim > 0.9999: match_count += 1
        
    # Content positions (9, 10, 11...)
    print(f"  Decoding Content (9+) using Tokenizer...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer_path = os.path.join(PROJECT_ROOT, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        text_content = "今天天气不错"
        content_ids = tokenizer.encode(text_content, add_special_tokens=False)
        print(f"  Tokenized '{text_content}': {content_ids}")
        
    except Exception as e:
        print(f"⚠️ Tokenizer load failed: {e}. Falling back to manual ID checking.")
        content_ids = [100644, 104307, 100832] # Fallback to known IDs
    
    # Map content ids to positions 9, 10, 11
    # TTS_EOS is at 9 + len(content_ids) = 9+3 = 12
    # Codec_BOS at 13
    
    for i in range(9, total_positions):
        target = official_embeds[0, i]
        
        # Determine Identity
        current_content_idx = i - 9
        
        if current_content_idx < len(content_ids):
            # It is content
            tid = content_ids[current_content_idx]
            cid = 2148 # C_PAD
            name = f"Content '{tokenizer.decode([tid])}' ({tid})"
            
            mine = text_table[tid] + codec_table[cid]
            sim = np.dot(target, mine) / (np.linalg.norm(target) * np.linalg.norm(mine) + 1e-9)
            
            check_mark = "✅" if sim > 0.9999 else "❌"
            print(f"    Pos {i} [{name}]: Sim={sim:.6f} {check_mark}")
            if sim > 0.9999: match_count += 1
            
        else:
            # Special Tokens after content
            # 12: TTS_EOS
            # 13: Codec_BOS
            
            candidates = {
                12: ("TTS_EOS", 151673, 2148),
                13: ("Codec_BOS", 151671, 2149),
            }
            
            if i in candidates:
                name, t, c = candidates[i]
                mine = text_table[t] + codec_table[c]
                sim = np.dot(target, mine) / (np.linalg.norm(target) * np.linalg.norm(mine) + 1e-9)
                check_mark = "✅" if sim > 0.9999 else "❌"
                print(f"    Pos {i} [{name}]: Sim={sim:.6f} {check_mark}")
                if sim > 0.9999: match_count += 1
            else:
                 print(f"    Pos {i} [Unexpected]: No mapping found.")

    print(f"\n[3/3] Conclusion: {match_count}/{total_positions} positions verified.")
    if match_count == total_positions:
        print("✅ FULL MATCH: We can reconstruct the prompt perfectly.")
    else:
        print("❌ MISMATCH: Reconstruction failed.")

if __name__ == "__main__":
    main()
