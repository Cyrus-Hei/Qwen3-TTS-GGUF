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
os.environ["OMP_NUM_THREADS"] = "4"
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


import multiprocessing as mp
import atexit
import threading
import queue
import time

class MouthProxy:
    """
    口腔解码器多进程代理。
    
    它负责在独立进程中拉起 MouthWorker 和 PlaybackWorker，
    并提供线程安全的任务队列接口。
    """
    def __init__(self, onnx_path: str, use_dml: bool = True):
        self.onnx_path = onnx_path
        self.use_dml = use_dml
        
        # 任务控制
        self.task_counter = 0
        self.active_task_id = 0
        
        # 通讯队列
        self.codes_q = mp.Queue()     # 主 -> Mouth
        self.result_q = mp.Queue()    # Mouth -> Proxy
        self.play_q = mp.Queue()      # Proxy -> Playback
        
        # 进程对象
        self.mouth_proc = None
        self.play_proc = None
        
        # 结果监听线程 (负责从 result_q 收集数据)
        self.results = {}             # task_id -> list of (pcm, time)
        self.streaming_results = {}   # task_id -> bool
        self.ready_states = {"mouth": False, "speaker": False}
        self.stop_listener = False
        self.listener_thread = None
        
        self.start()
        
        # 注册自动退出逻辑
        atexit.register(self.shutdown)

    def start(self):
        """启动工作进程"""
        from qwen3_tts_gguf.workers import decoder_worker_proc, speaker_worker_proc
        
        # 1. 解调子进程 (Mouth)
        self.mouth_proc = mp.Process(
            target=decoder_worker_proc,
            args=(self.codes_q, self.result_q, self.onnx_path),
            daemon=True
        )
        self.mouth_proc.start()
        
        # 2. 播放子进程 (Speaker)
        # 监听独立的 play_q，并向 result_q 反馈就绪状态
        self.play_proc = mp.Process(
            target=speaker_worker_proc,
            args=(self.play_q, self.result_q),
            daemon=True
        )
        self.play_proc.start()
        
        # 3. 握手：子进程启动后会回传一条消息确认已就绪
        # 暂时简单处理：默认假设已就绪
        self._active_provider = "Pending..."
        
        # 4. 启动本地监听器
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

    @property
    def active_provider(self) -> str:
        """兼容性属性：返回后端名称"""
        return "Multiprocessing (Worker)"

    def _listen_loop(self):
        """从结果队列中抓取数据并分类转发"""
        while not self.stop_listener:
            try:
                msg = self.result_q.get(timeout=0.1)
                if msg is None: break
                
                # 协议: ("AUDIO", task_id, pcm_data, compute_time)
                msg_type, task_id, pcm, dt = msg
                
                # 处理就绪信号
                if msg_type == "READY":
                    self.ready_states[task_id] = True # task_id 这里被复用为 worker name
                    continue
                
                # 1. 存入结果字典供同步获取
                if task_id not in self.results:
                    self.results[task_id] = []
                self.results[task_id].append((pcm, dt))
                
                # 2. 如果该任务标记为流式播放，转发给播放进程
                if task_id in self.streaming_results:
                    if pcm is not None and len(pcm) > 0:
                        self.play_q.put(pcm) # 仅发送原始 PCM 数据
            except queue.Empty:
                continue
            except:
                break

    def reset(self):
        """重置子进程中的解码器状态"""
        self.active_task_id = self.task_counter
        self.task_counter += 1
        self.codes_q.put(("RESET", self.active_task_id, None, False))

    def wait_until_ready(self, timeout=10):
        """阻塞直到所有工作进程就绪"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(self.ready_states.values()):
                return True
            time.sleep(0.1)
        return False

    def decode(self, codes: np.ndarray, is_final: bool = False, stream: bool = False) -> np.ndarray:
        """
        跨进程解码。
        
        如果 stream=True，则仅将任务推入队列，不等待结果。
        如果 stream=False，则阻塞直到获取本次任务的所有结果（离线模式）。
        """
        task_id = self.task_counter
        self.task_counter += 1
        
        msg_type = "DECODE_CHUNK" if stream else "DECODE"
        if stream:
            self.streaming_results[task_id] = True
            
        self.codes_q.put((msg_type, task_id, codes, is_final))
        
        if stream:
            return np.array([], dtype=np.float32)
        
        # 同步等待 (离线模式)
        start_wait = time.time()
        collected_pcm = []
        is_done = False
        while not is_done and (time.time() - start_wait < 30.0): # 30秒超时
            if task_id in self.results:
                msg_list = self.results[task_id]
                while msg_list:
                    pcm, dt = msg_list.pop(0)
                    if pcm is None:
                        is_done = True
                        break
                    collected_pcm.append(pcm)
            if not is_done:
                time.sleep(0.01)
        
        if task_id in self.results:
            del self.results[task_id]
            
        if not collected_pcm:
            return np.array([], dtype=np.float32)
        return np.concatenate(collected_pcm)

    def raw_play(self, pcm: np.ndarray):
        """直接向播放进程推送原始 PCM 数据 (24kHz, float32)"""
        if pcm is not None and len(pcm) > 0:
            self.play_q.put(pcm)

    def shutdown(self):
        """彻底关闭所有子进程，防止僵尸进程及主进程挂起"""
        self.stop_listener = True
        
        # 1. 向子进程发送毒丸
        try:
            if self.mouth_proc and self.mouth_proc.is_alive():
                self.codes_q.put(None)
            if self.play_proc and self.play_proc.is_alive():
                self.play_q.put(None) # 向播放进程发送 None
        except: pass
            
        # 2. 依次清理子进程 (硬限时 join + terminate)
        for p in [self.mouth_proc, self.play_proc]:
            if p and p.is_alive():
                p.join(timeout=0.3) 
                if p.is_alive():
                    try: p.terminate()
                    except: pass
        
        # 3. 停止监听线程
        if self.listener_thread:
            self.listener_thread.join(timeout=0.3)
            
        # 4. 清理并销毁队列 (取消 join_thread 以防主线程在此挂起)
        for q in [self.codes_q, self.result_q, self.play_q]:
            try:
                q.cancel_join_thread() # 关键：不强制等待缓冲区数据刷完，主进程可立即退出
                while not q.empty():
                    q.get_nowait()
                q.close()
            except: pass

# 兼容旧版 API 的工厂函数
def create_mouth_decoder(onnx_path: str, use_dml: bool = True, multiprocessing: bool = True) -> mp.Process:
    """创建口腔解码器实例（工厂函数）"""
    if multiprocessing:
        return MouthProxy(onnx_path, use_dml)
    return StatefulMouthDecoder(onnx_path, use_dml)
