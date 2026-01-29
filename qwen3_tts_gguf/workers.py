import os
import time
import numpy as np
import queue
import soundfile as sf

def wav_writer_proc(record_queue, filename, sample_rate=24000):
    abs_filename = os.path.abspath(filename)
    os.makedirs(os.path.dirname(abs_filename), exist_ok=True)
    try:
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
    except:
        abs_filename = abs_filename.replace(".wav", f"_{int(time.time())}.wav")
        f = sf.SoundFile(abs_filename, mode='w', samplerate=sample_rate, channels=1)
    try:
        while True:
            chunk = record_queue.get()
            if chunk is None: break
            if isinstance(chunk, str) and chunk == "CLEAR": continue
            f.write(chunk.flatten().astype(np.float32))
            f.flush()
    except: pass
    finally: f.close()

def decoder_worker_proc(codes_queue, pcm_queue, mouth_onnx_path, record_queue=None):
    """
    解码工人：使用新版 StatefulMouthDecoder 简化流式解码。
    支持 CLEAR 信号重置状态，支持 FINAL 信号输出剩余音频。
    
    消息协议:
    - CLEAR: 重置状态（新句子）
    - FINAL: 标记最后一个 chunk
    - (codes, is_final): 音频码 + 是否最后一帧
    - None: 退出
    """
    from qwen3_tts_gguf.mouth_decoder import StatefulMouthDecoder
    
    # 创建解码器实例
    decoder = StatefulMouthDecoder(mouth_onnx_path, use_dml=True)
    print(f"  [DecoderWorker] 已启动 (Provider: {decoder.active_provider})")
    
    try:
        while True:
            task = codes_queue.get()
            
            # 退出信号
            if task is None:
                pcm_queue.put(None)
                if record_queue: record_queue.put(None)
                break
            
            # 重置信号
            if isinstance(task, str) and task == "CLEAR":
                decoder.reset()
                if record_queue: record_queue.put("CLEAR")
                continue
            
            # 解析任务
            if isinstance(task, tuple) and len(task) == 2:
                codes, is_final = task
            else:
                # 兼容旧版：单独的 codes 数组默认不是 final
                codes = task
                is_final = False
            
            # 转换为 numpy
            working_codes = np.array(codes).astype(np.int64)
            
            if len(working_codes) == 0:
                continue
            
            # 调用新版解码器
            audio = decoder.decode(working_codes, is_final=is_final)
            
            # 交付有效音频
            if len(audio) > 0:
                pcm_queue.put(audio.copy())
                if record_queue: record_queue.put(audio.copy())

    except Exception as e:
        print(f"  [DecoderWorker] 异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pass


def speaker_worker_proc(pcm_queue, sample_rate=24000):
    import sounddevice as sd
    state = {"current_data": np.zeros((0, 1), dtype=np.float32), "started": False, "prefill": 4800}
    def audio_callback(outdata, frames, time_info, status):
        while True:
            try:
                new_item = pcm_queue.get_nowait()
                if new_item is None: break
                # 注意：Speaker 不处理 CLEAR，它只需顺序播放。
                # 它播完上一段的剩下的 PCM，自然会接着播下一段。
                state["current_data"] = np.concatenate([state["current_data"], new_item.reshape(-1, 1).astype(np.float32)], axis=0)
            except queue.Empty: break
        if not state["started"]:
            if len(state["current_data"]) >= state["prefill"]: state["started"] = True
            else: outdata.fill(0); return
        avail = len(state["current_data"])
        to_copy = min(avail, frames)
        if to_copy > 0:
            outdata[:to_copy] = state["current_data"][:to_copy]
            state["current_data"] = state["current_data"][to_copy:]
        if to_copy < frames:
            outdata[to_copy:].fill(0); state["started"] = False
    try:
        with sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=1024):
            while True: time.sleep(1)
    except: pass
