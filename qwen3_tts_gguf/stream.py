"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
import time
import numpy as np
from typing import Optional, List, Tuple, Union
from .constants import PROTOCOL, map_speaker, map_language
from .result import TTSResult, Timing, LoopOutput, TTSConfig
from .predictors.talker import TalkerPredictor
from .predictors.predictor import Predictor

from . import llama, logger
from .prompt_builder import PromptBuilder, PromptData

class TTSStream:
    """
    保存 Talker, Predictor, Decoder 记忆的语音流。
    """
    def __init__(self, engine, n_ctx=4096, voice_path: Optional[str] = None):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化推理核心 (Talker & Predictor)
        self.talker = TalkerPredictor(engine.talker_model, self.talker_ctx, self.talker_batch, self.assets)
        self.predictor = Predictor(engine.predictor_model, self.predictor_ctx, self.predictor_batch, self.assets)
        
        # 3. 音色锚点 (Voice)
        self.voice: Optional[TTSResult] = None
        if voice_path:
            self.set_voice_from_json(voice_path)
            
        self.decoder = getattr(engine, 'decoder', None)

    def _init_contexts(self):
        """初始化此语音流专属的推理环境"""
        logger.info(f"[Stream] 正在初始化独立 Context (n_ctx={self.n_ctx})...")
        
        # 使用 llama.py 升级版接口，确保参数安全
        self.talker_ctx = llama.create_context(self.engine.talker_model, n_ctx=self.n_ctx, embeddings=True)
        self.predictor_ctx = llama.create_context(self.engine.predictor_model, n_ctx=512, embeddings=False)
        
        self.talker_batch = llama.create_batch(self.n_ctx, embd_dim=2048)
        self.predictor_batch = llama.create_batch(32, embd_dim=1024)

    # =========================================================================
    # 核心推理 API (Adapt to Base / CustomVoice / VoiceDesign)
    # =========================================================================

    def clone(self, 
              text: str, 
              language: str = "chinese",
              config: Optional[TTSConfig] = None,
              verbose: bool = True) -> TTSResult:
        """
        [克隆模式] 对应 Base 模型能力。
        基于已设定的音色锚点（通过 set_voice 设定）合成新文本。
        """
        if self.voice is None:
            raise RuntimeError("⚠️ 请先调用 set_voice() 设定音色锚点，才能进行 clone。")
            
        cfg = config or TTSConfig()
        self.talker.clear_memory() # 确保记忆纯净
        
        # 克隆模式入口
        lang_id = map_language(language)
        pdata = PromptBuilder.build_clone_prompt(text, self.tokenizer, self.assets, self.voice, lang_id)
        
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        return self._post_process(text, pdata, lout, cfg=cfg)

    def custom(self,
               text: str,
               speaker: str,
               language: str = "chinese",
               instruct: Optional[str] = None,
               config: Optional[TTSConfig] = None,
               verbose: bool = True) -> TTSResult:
        """
        [精品音色模式] 对应 CustomVoice 模型能力。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        # 精品音色入口
        spk_id = map_speaker(speaker)
        lang_id = map_language(language) if language.lower() != "auto" else None
        pdata = PromptBuilder.build_custom_prompt(text, self.tokenizer, self.assets, spk_id, lang_id, instruct)
        
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        return self._post_process(text, pdata, lout, cfg=cfg)

    def design(self,
               text: str,
               instruct: str,
               language: str = "chinese",
               config: Optional[TTSConfig] = None,
               verbose: bool = True) -> TTSResult:
        """
        [音色设计模式] 对应 VoiceDesign 模型能力。
        """
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        
        # 设计模式入口
        lang_id = map_language(language) if language.lower() != "auto" else None
        pdata = PromptBuilder.build_design_prompt(text, self.tokenizer, self.assets, instruct, lang_id)
        
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        return self._post_process(text, pdata, lout, cfg=cfg)

    def tts(self, *args, **kwargs):
        """兼容性包装：现默认指向 clone 逻辑"""
        return self.clone(*args, **kwargs)


    def _run_engine_loop(self, pdata: PromptData, timing: Timing, cfg: TTSConfig, verbose: bool = False) -> LoopOutput:
        """核心推理循环：支持生成全量 Codes 并可选进行流式推送"""
        all_codes = []
        turn_summed_embeds = []
        chunk_buffer = []
        pushed_count = 0
        
        # 流式播放初始化
        if cfg.stream_play and self.decoder:
            self.decoder.reset()
            if verbose: logger.info(f"🚀 [Loop] 开始流式推送任务 (TaskID: {self.decoder.active_task_id})")
            
        for step_codes, summed_vec in self._run_engine_loop_gen(pdata, cfg, timing):
            all_codes.append(step_codes)
            turn_summed_embeds.append(summed_vec)
            
            # 流式分块推送
            if cfg.stream_play and self.decoder:
                chunk_buffer.append(step_codes)
                if len(chunk_buffer) >= cfg.decoder_chunk_size:
                    self.decoder.decode(np.array(chunk_buffer), is_final=False, stream=True)
                    pushed_count += len(chunk_buffer)
                    if verbose: print(f"\r   📡 已推送: {pushed_count:3} 帧 Codec...", end="", flush=True)
                    chunk_buffer = []

        # 结束流式推送
        if cfg.stream_play and self.decoder:
            self.decoder.decode(np.array(chunk_buffer) if chunk_buffer else np.zeros((0, 16)), is_final=True, stream=True)
            pushed_count += len(chunk_buffer)
            if verbose: print(f"\n   ✅ 推送完毕, 共计: {pushed_count} 帧。")

        return LoopOutput(all_codes=all_codes, summed_embeds=turn_summed_embeds, timing=timing)

    def _run_engine_loop_gen(self, pdata: PromptData, cfg: TTSConfig, timing: Timing):
        """
        [New] 生成器版推理循环。
        逐帧产出 (codes, summed_embeds)。
        """
        # Talker Prefill
        t_pre_s = time.time()
        m_hidden, m_logits = self.talker.prefill(pdata.embd, seq_id=0)
        timing.prefill_time = time.time() - t_pre_s
        
        for step_idx in range(cfg.max_steps):
            code_0 = self.engine._do_sample(
                m_logits, 
                do_sample=cfg.do_sample, 
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k
            )
            if code_0 == PROTOCOL["EOS"]:
                break
            
            # Predictor 补全
            t_c_s = time.time()
            step_codes, step_embeds_2048 = self.predictor.predict_frame(
                m_hidden, 
                code_0, 
                do_sample=cfg.sub_do_sample,
                temperature=cfg.sub_temperature,
                top_p=cfg.sub_top_p,
                top_k=cfg.sub_top_k
            )
            timing.predictor_loop_time += (time.time() - t_c_s)
            
            # Talker 反馈
            t_m_s = time.time()
            summed = np.sum(step_embeds_2048, axis=0) + self.assets.tts_pad.flatten()
            m_hidden, m_logits = self.talker.decode_step(summed, seq_id=0)
            timing.talker_loop_time += (time.time() - t_m_s)
            
            yield step_codes, summed
            
        else:
            print(f"\n⚠️  [Stream] 推理达到了最大步数限制 ({cfg.max_steps}) 仍未检出 EOS。")
            
        timing.total_steps = len(pdata.embd) + step_idx

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     lout: LoopOutput,
                     cfg: Optional[TTSConfig] = None) -> TTSResult:
        """
        后处理：生成音频波形并封装。
        """
        audio = None
        if self.decoder:
            t0 = time.time()
            audio = self.decoder.decode(np.array(lout.all_codes), is_final=True, stream=False)
            lout.timing.decoder_render_time = time.time() - t0

        return TTSResult(
            audio=audio,
            text=text,
            text_ids=pdata.text_ids,
            spk_emb=pdata.spk_emb,
            codes=np.array(lout.all_codes),
            summed_embeds=lout.summed_embeds,
            stats=lout.timing
        )

    def save_audio(self, audio: np.ndarray, path: str):
        """兼容层：保存音频文件"""
        import soundfile as sf
        sf.write(path, audio, 24000)

    def play_audio(self, audio: np.ndarray):
        """兼容层：播放音频"""
        import sounddevice as sd
        sd.play(audio, 24000)
        sd.wait()

    def reset(self):
        """重置流：清除记忆与音色设置"""
        self.talker_ctx.kv_cache_clear()
        self.predictor_ctx.kv_cache_clear()
        self.voice = None
        logger.info("扫 [Stream] 记忆与音色已清除。")

    def shutdown(self):
        """释放占用的资源"""
        try:
            llama.llama_batch_free(self.talker_batch)
            llama.llama_batch_free(self.predictor_batch)
            llama.llama_free(self.talker_ctx)
            llama.llama_free(self.predictor_ctx)
        except: pass

    def __del__(self):
        self.shutdown()

    # =========================================================================
    # 音色设置 API (Voice Management)
    # =========================================================================

    def set_voice(self, source: Union[TTSResult, str], text: Optional[str] = None):
        """统一设置当前流的音色锚点。"""
        if isinstance(source, TTSResult):
            self._set_voice_from_result(source)
        elif isinstance(source, str):
            low_source = source.lower()
            if low_source.endswith(".json"):
                self._set_voice_from_json(source)
            elif low_source.endswith((".wav", ".mp3", ".flac")):
                self._set_voice_from_wav(source, text or "")
            else:
                self.set_voice_from_speaker(source, text or "你好")
        else:
            raise TypeError(f"Unsupported voice source type: {type(source)}")

    def _set_voice_from_result(self, res: TTSResult):
        """直接将一个 TTSResult 设为当前流的音色锚点。"""
        if not res.is_valid_anchor:
            raise ValueError("Provided TTSResult is not a valid anchor.")
        self.voice = res
        logger.info(f"🎭 Voice switched to: {res.text[:20]}...")

    def _set_voice_from_json(self, path: str):
        """从 JSON 文件恢复音色锚点"""
        res = TTSResult.from_json(path)
        self._set_voice_from_result(res)

    def _set_voice_from_wav(self, wav_path: str, text: str):
        """克隆音色：从外部文件提取特征并设为音色锚点"""
        if self.engine.encoder is None:
            raise RuntimeError("⚠️ 编码器模块未加载，无法执行音色克隆。")
        logger.info(f"🎤 Extracting features from: {wav_path}")
        codes, spk_emb = self.engine.encoder.encode(wav_path)
        res = TTSResult(text=text, text_ids=self.tokenizer.encode(text).ids, spk_emb=spk_emb, codes=codes)
        self._set_voice_from_result(res)

    def set_voice_from_speaker(self, speaker_id: str, text: str, language: str = "chinese", config: Optional[TTSConfig] = None, verbose: bool = False) -> TTSResult:
        """从指定内置说话人生成一个音色锚点并激活"""
        logger.info(f"📍 Initializing Voice from Speaker: {speaker_id}")
        cfg = config or TTSConfig()
        self.talker.clear_memory()
        lang_id = map_language(language)
        spk_id = map_speaker(speaker_id)
        pdata = PromptBuilder.build_custom_prompt(text, self.tokenizer, self.assets, spk_id, lang_id)
        timing = Timing()
        timing.prompt_time = pdata.compile_time
        lout = self._run_engine_loop(pdata, timing, cfg, verbose=verbose)
        res = self._post_process(text, pdata, lout)
        self._set_voice_from_result(res)
        return res
