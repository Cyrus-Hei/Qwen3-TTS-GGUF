from transformers.cache_utils import DynamicCache
import torch

def debug_cache_struct():
    cache = DynamicCache()
    # 模拟更新
    k = torch.randn(1, 16, 5, 64)
    v = torch.randn(1, 16, 5, 64)
    cache.update(k, v, 0)
    
    # 尝试使用 crop 方法
    if hasattr(cache, 'crop'):
        print(f"正在测试 crop(3)...")
        cache.crop(3)
        print(f"裁剪后长度: {cache.get_seq_length()}")

if __name__ == "__main__":
    debug_cache_struct()
