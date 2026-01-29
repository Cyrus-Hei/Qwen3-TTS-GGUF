"""
StatefulMouthDecoder - 状态化口腔解码器封装
提供友好的对外接口，内部自动管理 KV Cache 和历史状态。

使用方法:
    decoder = StatefulMouthDecoder("model/qwen3_tts_decoder_stateful.onnx")
    
    # 流式调用
    for codes in code_stream:
        is_final = (codes is last_chunk)
        audio = decoder.decode(codes, is_final=is_final)
        play(audio)
    
    # 重置状态（新句子）
    decoder.reset()
"""
import os
import numpy as np

class StatefulMouthDecoder:
    """
    状态化 ONNX 嘴巴解码器封装。
    
    特点:
    - 内部自动管理 KV Cache 和历史状态
    - 支持流式和一次性解码
    - 外部只需传递 audio_codes 和 is_final
    - 自动处理 DirectML 加速（如果可用）
    """
    
    # 常量配置
    NUM_LAYERS = 8
    NUM_HEADS = 16
    HEAD_DIM = 64
    PRE_CONV_WINDOW = 2
    CONV_HISTORY_WINDOW = 4
    LOOKAHEAD_FRAMES = 4
    KV_CACHE_WINDOW = 72
    SAMPLES_PER_FRAME = 1920
    
    def __init__(self, onnx_path: str, use_dml: bool = True):
        """
        初始化解码器。
        
        Args:
            onnx_path: 状态化 ONNX 模型路径
            use_dml: 是否尝试使用 DirectML 加速 (Windows GPU)
        """
        import onnxruntime as ort
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")
        
        # 选择 Provider
        providers = ['CPUExecutionProvider']
        if use_dml:
            available = ort.get_available_providers()
            if 'DmlExecutionProvider' in available:
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.output_names = [out.name for out in self.sess.get_outputs()]
        
        # 获取实际使用的 provider
        self.active_provider = self.sess.get_providers()[0]
        
        # 初始化状态
        self.reset()
    
    def reset(self):
        """重置所有内部状态（开始新的句子时调用）"""
        self.pre_conv_history = np.zeros((1, 512, 0), dtype=np.float32)
        self.latent_buffer = np.zeros((1, 1024, 0), dtype=np.float32)
        self.conv_history = np.zeros((1, 1024, 0), dtype=np.float32)
        
        # KV Cache: [NUM_LAYERS] 个 Key + [NUM_LAYERS] 个 Value
        self.past_keys = [
            np.zeros((1, self.NUM_HEADS, 0, self.HEAD_DIM), dtype=np.float32)
            for _ in range(self.NUM_LAYERS)
        ]
        self.past_values = [
            np.zeros((1, self.NUM_HEADS, 0, self.HEAD_DIM), dtype=np.float32)
            for _ in range(self.NUM_LAYERS)
        ]
        
        # 累计帧数（用于精准切割）
        self.total_frames_processed = 0
        self.total_samples_output = 0
    
    def decode(self, audio_codes: np.ndarray, is_final: bool = False) -> np.ndarray:
        """
        解码音频码为波形。
        
        Args:
            audio_codes: 形状 [N, 16] 的音频码（N 为帧数，16 为量化器数）
            is_final: 是否是最后一个 chunk（最后一个 chunk 会输出所有剩余音频）
        
        Returns:
            audio: 形状 [num_samples] 的音频波形 (float32, 范围 [-1, 1])
        """
        # 输入规范化
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[np.newaxis, ...]  # [1, N, 16]
        
        audio_codes = audio_codes.astype(np.int64)
        n_frames = audio_codes.shape[1]
        
        if n_frames == 0:
            return np.array([], dtype=np.float32)
        
        # 构建输入 feed dict
        feed = {
            "audio_codes": audio_codes,
            "is_last": np.array([1.0 if is_final else 0.0], dtype=np.float32),
            "pre_conv_history": self.pre_conv_history,
            "latent_buffer": self.latent_buffer,
            "conv_history": self.conv_history,
        }
        for i in range(self.NUM_LAYERS):
            feed[f"past_key_{i}"] = self.past_keys[i]
            feed[f"past_value_{i}"] = self.past_values[i]
        
        # 执行推理
        outputs = self.sess.run(self.output_names, feed)
        
        # 解包输出
        final_wav = outputs[0]        # [1, num_samples]
        valid_samples = int(outputs[1][0])  # 有效样本数
        
        # 更新状态
        self.pre_conv_history = outputs[2]
        self.latent_buffer = outputs[3]
        self.conv_history = outputs[4]
        for i in range(self.NUM_LAYERS):
            self.past_keys[i] = outputs[5 + i]
            self.past_values[i] = outputs[5 + self.NUM_LAYERS + i]
        
        # 提取有效音频
        audio = final_wav[0, :valid_samples] if valid_samples > 0 else np.array([], dtype=np.float32)
        
        # 更新统计
        self.total_frames_processed += n_frames
        self.total_samples_output += len(audio)
        
        return audio.astype(np.float32)
    
    def decode_full(self, audio_codes: np.ndarray) -> np.ndarray:
        """
        一次性解码所有音频码（非流式场景）。
        
        Args:
            audio_codes: 形状 [N, 16] 的完整音频码序列
        
        Returns:
            audio: 完整的音频波形
        """
        self.reset()
        return self.decode(audio_codes, is_final=True)
    
    @property
    def info(self) -> dict:
        """返回解码器状态信息"""
        return {
            "provider": self.active_provider,
            "total_frames": self.total_frames_processed,
            "total_samples": self.total_samples_output,
            "kv_cache_len": self.past_keys[0].shape[2] if self.past_keys else 0,
        }


# 兼容旧版 API 的工厂函数
def create_mouth_decoder(onnx_path: str, use_dml: bool = True) -> StatefulMouthDecoder:
    """创建口腔解码器实例（工厂函数）"""
    return StatefulMouthDecoder(onnx_path, use_dml)
