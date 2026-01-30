import ctypes
import os

def list_all_llama_funcs():
    dll_path = "qwen3_tts_gguf/bin/llama.dll"
    os.chdir("qwen3_tts_gguf/bin")
    ctypes.CDLL("./ggml.dll")
    ctypes.CDLL("./ggml-base.dll")
    # On Windows, we can't easily iterate exports via ctypes.
    # We might need to use a tool or just check common ones.
    lib = ctypes.CDLL("./llama.dll")
    
    # Let's try some other common ones
    checks = [
        "llama_kv_cache_seq_rm",
        "llama_kv_cache_tokens_rm",
        "llama_kv_cache_rm",
        "llama_kv_cache_clear",
        "llama_kv_cache_seq_keep",
        "llama_kv_cache_seq_shift",
        "llama_kv_cache_defrag",
        "llama_get_kv_cache_token_count"
    ]
    
    for c in checks:
        try:
            getattr(lib, c)
            print(f"✅ Found: {c}")
        except:
            pass

if __name__ == "__main__":
    list_all_llama_funcs()
