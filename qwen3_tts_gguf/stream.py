"""
stream.py - TTS 语音流
核心逻辑所在，管理单次会话的上下文，支持流式和非流式合成。
"""
import time
import numpy as np
from typing import Optional, List, Tuple
from .constants import PROTOCOL, SPEAKER_MAP, LANGUAGE_MAP
from .result import TTSResult, Timing
from .predictors.master import MasterPredictor
from .predictors.craftsman import CraftsmanPredictor

from . import llama, logger
from .identity import VoiceIdentity
from .prompt_builder import PromptBuilder, PromptData

class TTSStream:
    """
    保存大师、工匠、嘴巴记忆的语音流。
    每个流拥有独立的 llama Context 和 ChatManager。
    """
    def __init__(self, engine, speaker_id, language, n_ctx=4096):
        self.engine = engine
        self.assets = engine.assets
        self.tokenizer = engine.tokenizer
        self.n_ctx = n_ctx
        
        # 映射 ID
        self.spk_id = self._map_speaker(speaker_id)
        self.lang_id = self._map_language(language)
        
        # 1. 初始化流独立的 Context 和 Batch
        self._init_contexts()
        
        # 2. 初始化记忆管理器 (需在 master 之后，因为它依赖 master.cur_pos)
        self.master = MasterPredictor(engine.m_model, self.m_ctx, self.m_batch, self.assets)
        self.craftsman = CraftsmanPredictor(engine.c_model, self.c_ctx, self.c_batch, self.assets)
        
        # 3. 初始化身份锚点 (Identity)
        self.identity = VoiceIdentity()
        
        self.mouth = getattr(engine, 'mouth', None)
        # 状态标志
        self.is_first_sentence = True

    def _init_contexts(self):
        """初始化此语音流专属的推理环境"""
        logger.info(f"[Stream] 正在初始化独立 Context (n_ctx={self.n_ctx})...")
        
        # 大师模型参数
        m_params = llama.llama_context_default_params()
        m_params.n_ctx = self.n_ctx
        m_params.embeddings = True
        self.m_ctx = llama.llama_init_from_model(self.engine.m_model, m_params)
        
        # 工匠模型参数 (固定 512)
        c_params = llama.llama_context_default_params()
        c_params.n_ctx = 512
        c_params.embeddings = False
        self.c_ctx = llama.llama_init_from_model(self.engine.c_model, c_params)
        
        if not self.m_ctx or not self.c_ctx:
            raise RuntimeError("llama Context 初始化失败 (显存不足或模型错误)")
            
        # 批次缓冲区
        self.m_batch = llama.llama_batch_init(self.n_ctx, 2048, 1)
        self.c_batch = llama.llama_batch_init(32, 1024, 1)

    def tts(self, 
            text: str, 
            play: bool = False, 
            save_path: Optional[str] = None,
            temperature: float = 0.5, 
            max_steps: int = 600,
            verbose: bool = True) -> TTSResult:
        """
        高层接口：协调 Prompt 构建、核心推理与后处理。
        """
        # 1. 准备数据与 Prompt
        t_p_start = time.time()
        if self.identity.is_set:
            pdata = PromptBuilder.build_clone_prompt(
                text, self.identity, self.tokenizer, self.assets, self.lang_id
            )
        else:
            pdata = PromptBuilder.build_native_prompt(
                text, self.tokenizer, self.assets, self.lang_id, self.spk_id
            )
        prompt_time = time.time() - t_p_start

        # 2. 驱动内核进行生成 (核心循环)
        all_codes, turn_summed_embeds, timing = self._run_engine_loop(
            pdata.embd, temperature, max_steps
        )
        timing.prompt_time = prompt_time

        # 3. 后处理 (音频渲染与结果封装)
        res = self._post_process(text, pdata, all_codes, turn_summed_embeds, timing)

        # 4. 执行副作用 (播放/保存)
        self._execute_side_effects(res, play, save_path)
        
        return res

    def _run_engine_loop(self, 
                       prompt_embeds: np.ndarray, 
                       temperature: float, 
                       max_steps: int) -> Tuple[List[List[int]], List[np.ndarray], Timing]:
        """
        内核层：负责 Master 与 Craftsman 的逐帧推理循环。
        """
        timing = Timing()
        self.master.clear_memory()
        self.mouth.reset()
        
        all_codes = []
        turn_summed_embeds = []

        # 大师 Prefill
        t_pre_s = time.time()
        m_hidden, m_logits = self.master.prefill(prompt_embeds, seq_id=0)
        timing.prefill_time = time.time() - t_pre_s
        
        for step_idx in range(max_steps):
            code_0 = self.engine._do_sample(m_logits, temperature)
            if code_0 == PROTOCOL["EOS"]:
                m_hidden, m_logits = self.master.decode_step(
                    self.assets.emb_tables[0][PROTOCOL["EOS"]].flatten() + self.assets.tts_pad.flatten(),
                    seq_id=0
                )
                break
            
            # 工匠补全
            t_c_s = time.time()
            step_codes, step_embeds_2048 = self.craftsman.predict_frame(m_hidden, code_0, temperature=temperature)
            timing.craftsman_loop_time += (time.time() - t_c_s)
            
            all_codes.append(step_codes)
            
            # 大师反馈
            t_m_s = time.time()
            summed = np.sum(step_embeds_2048, axis=0) + self.assets.tts_pad.flatten()
            turn_summed_embeds.append(summed.copy())
            m_hidden, m_logits = self.master.decode_step(summed, seq_id=0)
            timing.master_loop_time += (time.time() - t_m_s)
            
        timing.total_steps = len(all_codes)
        return all_codes, turn_summed_embeds, timing

    def _post_process(self, 
                     text: str, 
                     pdata: PromptData, 
                     all_codes: List[List[int]], 
                     summed_embeds: List[np.ndarray], 
                     timing: Timing) -> TTSResult:
        """
        后处理层：渲染音频并封装 TTSResult。
        """
        t_r_s = time.time()
        audio_out = self.mouth.decode_full(np.array(all_codes))
        timing.mouth_render_time = time.time() - t_r_s

        res = TTSResult(
            audio=audio_out,
            text=text,
            text_ids=pdata.text_ids,
            spk_emb=pdata.spk_emb,
            codes=np.array(all_codes),
            summed_embeds=summed_embeds,
            stats=timing
        )
        return res

    def _execute_side_effects(self, res: TTSResult, play: bool, save_path: str):
        """执行保存和播放等副作用"""
        if save_path:
            import soundfile as sf
            sf.write(save_path, res.audio, 24000)
            
        if play:
            import sounddevice as sd
            sd.play(res.audio, 24000)
            sd.wait()

    def reset(self):
        """完全重置身份和状态"""
        self.identity.text = None
        self.identity.text_ids = None
        self.identity.spk_emb = None
        self.identity.codes = None
        self.identity.summed_embeds = None
        self.master.clear_memory()
        self.is_first_sentence = True
        self.mouth.reset()

    def shutdown(self):
        """释放此流占用的本地 Context 和 Batch 资源"""
        logger.info("[Stream] 正在释放资源...")
        try:
            llama.llama_batch_free(self.m_batch)
            llama.llama_batch_free(self.c_batch)
            llama.llama_free(self.m_ctx)
            llama.llama_free(self.c_ctx)
        except: pass

    def __del__(self):
        self.shutdown()

    def set_identity(self, text: str, audio: Optional[np.ndarray] = None, speaker_id: Optional[str] = None):
        """
        核心接口一：固定音色锚点。
        """
        if audio is not None:
            # TODO: 加载编码器后的提取逻辑
            logger.info("Setting identity via reference audio (Encoder needed)...")
        else:
            self.set_identity_from_speaker(speaker_id or self.spk_id, self.lang_id, text, play=False)

    def set_identity_from_speaker(self, 
                                 speaker_id: str, 
                                 language: str, 
                                 text: str, 
                                 play: bool = True):
        """
        指定一个 speaker_id 和一段文本，合成后将其作为本流的永恒音色锚点。
        """
        logger.info(f"📍 Setting Identity from Speaker: {speaker_id}, Language: {language}")
        
        # 暂时修改流的状态以适配本次生成
        old_spk = self.spk_id
        old_lang = self.lang_id
        self.spk_id = self._map_speaker(speaker_id)
        self.lang_id = self._map_language(language)
        
        # 强制清除旧身份
        self.identity.text = None
        self.identity.text_ids = None
        self.identity.spk_emb = None
        self.identity.codes = None
        self.identity.summed_embeds = None
        
        # 执行合成 (获取结果并手动设置锚点)
        res = self.tts(text, play=play, verbose=False)
        self.identity.set_identity(res.text, res.text_ids, res.spk_emb, res.codes, res.summed_embeds)
        
        # 恢复流的默认偏好
        self.spk_id = old_spk
        self.lang_id = old_lang
        
        return res

    def _map_speaker(self, spk):
        if isinstance(spk, int): return spk
        return SPEAKER_MAP.get(str(spk).lower(), 3065)

    def _map_language(self, lang):
        if isinstance(lang, int): return lang
        return LANGUAGE_MAP.get(str(lang).lower(), 2055)
