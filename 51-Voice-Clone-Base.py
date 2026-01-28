import os
import sys
import time
import torch
import soundfile as sf
import subprocess

# Add current directory to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "Qwen3-TTS")
sys.path.insert(0, SOURCE_DIR) # Prioritize local source

import qwen_tts
print(f"Imported qwen_tts from: {qwen_tts.__file__}")

from qwen_tts import Qwen3TTSModel

def play_audio(file_path):
    print(f"Playing {file_path}...")
    # Use powershell to play audio
    try:
        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync();"], check=True)
    except Exception as e:
        print(f"Failed to play audio: {e}")

def main():
    # Use CPU or CUDA if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Local model path - Base model for voice cloning
    MODEL_PATH = os.path.abspath("Qwen3-TTS-12Hz-1.7B-Base")

    # Reference audio and text
    ref_audio = os.path.abspath("output/sample.wav")
    ref_text = "你好，我是具有随机性的千问3-TTS，这是我的终极进化形态"  # 请根据实际音频内容修改

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found: {MODEL_PATH}")
        print(f"Please download the model first using:")
        print(f"  pip install -U modelscope")
        print(f"  modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base")
        return

    # Check if reference audio exists
    if not os.path.exists(ref_audio):
        print(f"Error: Reference audio not found: {ref_audio}")
        return

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Reference audio: {ref_audio}")

    try:
        # Measure Model Loading Time
        print("Starting model load...")

        # Define dtype
        dtype = torch.float32 if device == "cpu" else torch.bfloat16

        t_load_start = time.time()
        tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map=device,
            dtype=dtype,
            # attn_implementation="flash_attention_2" # Comment out if no flash attn support on Windows/CPU
        )
        t_load_end = time.time()
        load_time = t_load_end - t_load_start
        print(f"Model loaded in {load_time:.4f} seconds.")

        # Text to synthesize
        text = "你好！我是用 Qwen3-TTS Base 模型进行声音克隆的示例。请听我说话的音色是否与参考音频一致。"
        output_file = "51_voice_clone_output.wav"

        print(f"\nGenerating audio for text: '{text}'")
        print(f"Reference text: '{ref_text}'")

        # Measure Inference Time
        t_infer_start = time.time()
        wavs, sr = tts.generate_voice_clone(
            text=text,
            language="Chinese",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        t_infer_end = time.time()
        infer_time = t_infer_end - t_infer_start
        print(f"Inference completed in {infer_time:.4f} seconds.")

        # Save audio
        sf.write(output_file, wavs[0], sr)
        print(f"Audio saved to: {output_file}")

        # Play audio
        print("\nPlaying generated audio...")
        play_audio(output_file)

        # Compare with reference audio
        print("\nPlaying reference audio for comparison...")
        play_audio(ref_audio)

    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
