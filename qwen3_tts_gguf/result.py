"""
result.py - 合成结果封装类
包含音频波形、核心推理要素以及详细的性能统计信息。
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .constants import SAMPLE_RATE

@dataclass
class Timing:
    """性能耗时统计"""
    prompt_time: float = 0.0
    prefill_time: float = 0.0
    master_loop_time: float = 0.0
    craftsman_loop_time: float = 0.0
    mouth_render_time: float = 0.0
    total_steps: int = 0

    @property
    def total_inference_time(self) -> float:
        return (self.prompt_time + self.prefill_time + 
                self.master_loop_time + self.craftsman_loop_time + 
                self.mouth_render_time)

@dataclass
class LoopOutput:
    """推理内核循环的输出封装"""
    all_codes: List[List[int]]     # 所有生成的 Codec IDs
    summed_embeds: List[np.ndarray] # 叠加特征序列
    timing: Timing                  # 性能统计对象

@dataclass
class GenConfig:
    """推理控制参数封装"""
    temperature: float = 0.5
    max_steps: int = 600
    top_p: float = 1.0
    top_k: int = 50

@dataclass
class TTSResult:
    """TTS 合成全量结果"""
    audio: np.ndarray              # 音频数据 (PCM float32)
    text: str                      # 输入文本
    text_ids: List[int]            # 文本 Token IDs
    spk_emb: np.ndarray            # 全局音色向量 (2048)
    codes: np.ndarray              # 音频 Codec IDs (T, 16)
    summed_embeds: List[np.ndarray]# 音频叠加特征 (T, 2048)
    stats: Timing                  # 性能统计对象
    
    @property
    def duration(self) -> float:
        """音频时长 (s)"""
        return len(self.audio) / SAMPLE_RATE if len(self.audio) > 0 else 0
    
    @property
    def rtf(self) -> float:
        """实时因子 (Real-Time Factor)"""
        if self.duration == 0: return 0
        return self.stats.total_inference_time / self.duration

    def print_stats(self):
        """打印性能报告"""
        s = self.stats
        print("-" * 40)
        print(f"性能分析报告 (音频长度: {self.duration:.2f}s | 文本长度: {len(self.text)})")
        print(f"  1. Prompt 编译:   {s.prompt_time:.4f}s")
        print(f"  2. 大师 Prefill:  {s.prefill_time:.4f}s")
        print(f"  3. 自回环总计:    {s.master_loop_time + s.craftsman_loop_time:.4f}s")
        print(f"     └─ 大师 (Master):    {s.master_loop_time:.4f}s")
        print(f"     └─ 工匠 (Craftsman): {s.craftsman_loop_time:.4f}s")
        print(f"  4. 嘴巴渲染 (Mouth): {s.mouth_render_time:.4f}s")
        print("-" * 40)
        print(f"总耗时: {s.total_inference_time:.2f}s | RTF: {self.rtf:.2f}")
