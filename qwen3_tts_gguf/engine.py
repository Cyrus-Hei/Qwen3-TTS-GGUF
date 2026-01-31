"""
engine.py - Qwen3-TTS 核心引擎
负责资源管理、模型初始化、音频特征提取及渲染。
"""
import os
import ctypes
from typing import Optional, List, Tuple
from . import llama, logger
from .assets import AssetsManager
from .stream import TTSStream
from .sampler import sample
from .decoder import create_decoder
from .predictors.encoder import EncoderPredictor

class TTSEngine:
    """
    Qwen3-TTS 引擎：资源池与 Stream 工厂。
    """
    def __init__(self, model_dir="model", verbose=True):
        import time
        import numpy as np
        from tokenizers import Tokenizer
        
        t_start = time.time()
        self.project_root = os.getcwd()
        self.model_dir = os.path.join(self.project_root, model_dir)
        
        # 路径定义 (全线标准化命名)
        self.paths = {
            "talker_gguf": os.path.join(self.model_dir, "qwen3_tts_talker.gguf"),
            "predictor_gguf": os.path.join(self.model_dir, "qwen3_tts_predictor.gguf"),
            "decoder_onnx": os.path.join(self.model_dir, "qwen3_tts_decoder.onnx"),
            "codec_enc_onnx": os.path.join(self.model_dir, "qwen3_tts_codec_encoder.onnx"),
            "spk_enc_onnx": os.path.join(self.model_dir, "qwen3_tts_speaker_encoder.onnx"),
            "tokenizer": os.path.join(self.model_dir, 'tokenizer', 'tokenizer.json'),
        }
        
        # 1. 资产加载
        t1 = time.time()
        self.assets = AssetsManager(self.model_dir)
        
        # 使用轻量级 tokenizers 库加载
        if not os.path.exists(self.paths['tokenizer']):
            raise FileNotFoundError(f"未找到轻量级分词文件: {self.paths['tokenizer']}。")
            
        self.tokenizer = Tokenizer.from_file(self.paths['tokenizer'])
        if verbose: print(f"📦 [Engine] 资产与词表加载完成 (耗时: {time.time()-t1:.2f}s)")
        
        # 2. 模型引擎初始化
        t2 = time.time()
        self._init_llama_engines()
        if verbose: print(f"🧠 [Engine] GGUF 推理后端就绪 (耗时: {time.time()-t2:.2f}s)")
        
        # 3. 解码器渲染模块 (多进程模式)
        t3 = time.time()
        self.decoder = create_decoder(self.paths["decoder_onnx"], use_dml=True)
        # 等待子进程就绪
        if verbose: print("⏳ [Engine] 正在拉起多进程渲染链条并等待就绪...")
        ready = self.decoder.wait_until_ready(timeout=10)
        if not ready:
            print("⚠️ [Engine] 渲染进程负载较慢，部分模块(Decoder/Speaker)未能在10s内响应就绪信号。")
        else:
            if verbose: print(f"✅ [Engine] 渲染链条就绪: Decoder {self.decoder.ready_states['decoder']} | Speaker {self.decoder.ready_states['speaker']} (耗时: {time.time()-t3:.2f}s)")
        
        # 4. 音频及说话人编码器
        t4 = time.time()
        self.encoder = None
        if os.path.exists(self.paths["codec_enc_onnx"]) and os.path.exists(self.paths["spk_enc_onnx"]):
            self.encoder = EncoderPredictor(self.paths["codec_enc_onnx"], self.paths["spk_enc_onnx"], use_dml=False)
            if verbose: print(f"🎤 [Engine] 编码器加载完成 (耗时: {time.time()-t4:.2f}s)")
        else:
            print("ℹ️ [Engine] 未找到编码器模型，音色克隆功能将不可用。")
            
        print(f"🚀 [Engine] 引擎全链路初始化完成! 总耗时: {time.time()-t_start:.2f}s")

    def _init_llama_engines(self):
        """初始化 GGUF 模型（仅加载模型，不创建 Context）"""
        logger.info("[Engine] 正在加载 GGUF 模型...")
        
        t_path = os.path.relpath(self.paths["talker_gguf"], os.getcwd())
        p_path = os.path.relpath(self.paths["predictor_gguf"], os.getcwd())
        
        self.talker_model = llama.load_model(t_path, n_gpu_layers=-1)
        self.predictor_model = llama.load_model(p_path, n_gpu_layers=-1)
        
        if not self.talker_model or not self.predictor_model:
            raise RuntimeError("GGUF 模型加载失败，请检查路径。")
            
        logger.info("✅ [Engine] GGUF 模型加载完成。")

    def create_stream(self, n_ctx=4096, voice_path: Optional[str] = None) -> TTSStream:
        """工厂方法：创建语音流"""
        return TTSStream(self, n_ctx=n_ctx, voice_path=voice_path)

    def _do_sample(self, logits, do_sample=True, temperature=0.5, top_p=1.0, top_k=50):
        """引擎内部采样辅助 (供 Stream 调用)"""
        if not do_sample:
            return int(np.argmax(logits))
        from .sampler import sample
        return sample(logits, temperature=temperature, top_p=top_p, top_k=top_k)

    def shutdown(self):
        """释放资源"""
        logger.info("[Engine] 正在关闭引擎...")
        try:
            if hasattr(self, 'decoder') and hasattr(self.decoder, 'shutdown'):
                self.decoder.shutdown()
            llama.llama_model_free(self.talker_model)
            llama.llama_model_free(self.predictor_model)
        except: pass
        logger.info("✅ [Engine] 引擎持有的模型资源已释放。")

    def __del__(self):
        self.shutdown()
