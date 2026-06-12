import os
import sys
import subprocess

# --- CONFIGURATION ---
# Target folder containing audio files to transcribe
TARGET_DIR = os.path.join(os.getcwd(), "voices")

# Paths based on your specific directory layout
WHISPER_DIR = os.path.join(os.getcwd(), "whisper.cpp")
MODEL_PATH = os.path.join(WHISPER_DIR, "whisper-large-v3-turbo-q8_0.gguf")

# Supported input audio formats
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")
# ---------------------

def find_whisper_executable():
    """Locates the whisper executable regardless of zip release variation."""
    possible_names = ["whisper-cli.exe", "main.exe", "whisper-cli", "main"]
    for name in possible_names:
        exe_path = os.path.join(WHISPER_DIR, name)
        if os.path.exists(exe_path):
            return exe_path
    return None

def main():
    print("\n🔮 Qwen3-TTS Whisper.cpp Batch Transcription Wrapper")
    print("=" * 55)

    # 1. Validation checks
    if not os.path.exists(TARGET_DIR):
        print(f"❌ Target directory not found: {TARGET_DIR}")
        return

    whisper_exe = find_whisper_executable()
    if not whisper_exe:
        print(f"❌ Could not find whisper-cli.exe or main.exe inside: {WHISPER_DIR}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Whisper model file not found at: {MODEL_PATH}")
        return

    print(f"⚙️  Using Executable: {os.path.basename(whisper_exe)}")
    print(f"⚙️  Using Model:      {os.path.basename(MODEL_PATH)}")
    print(f"⚙️  Scanning Folder:  {TARGET_DIR}\n")

    # 2. Collect files that require processing
    all_files = os.listdir(TARGET_DIR)
    audio_files = [f for f in all_files if f.lower().endswith(AUDIO_EXTENSIONS)]

    if not audio_files:
        print("🎉 No audio files found in the target directory to process.")
        return

    processed_count = 0
    skipped_count = 0

    # 3. Process files
    for filename in audio_files:
        basename, _ = os.path.splitext(filename)
        audio_path = os.path.join(TARGET_DIR, filename)
        txt_path = os.path.join(TARGET_DIR, f"{basename}.txt")

        # Skip files that already have a transcript to avoid wasting GPU cycles
        if os.path.exists(txt_path):
            skipped_count += 1
            continue

        print(f"🎙️ Transcribing [{filename}]...")

        # Build whisper.cpp command array (-nt skips printing structural timestamps)
        cmd = [
            whisper_exe,
            "-m", MODEL_PATH,
            "-f", audio_path,
            "-nt",
            "-l", "auto"
        ]

        # Enforce execution on GPU device 1 matching your TTS setup
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"

        try:
            # whisper.cpp pipes logs to stderr and actual parsed text to stdout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding="utf-8", 
                env=env,
                check=True
            )
            
            clean_text = result.stdout.strip()

            if clean_text:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)
                print(f"📄 Generated -> {basename}.txt (\"{clean_text[:40]}...\")")
                processed_count += 1
            else:
                print(f"⚠️ Warning: Whisper returned empty string for {filename}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed processing {filename}. Error details:\n{e.stderr}")
        except Exception as ex:
            print(f"❌ System error during processing: {str(ex)}")

    print("=" * 55)
    print(f"🏁 Batch Processing Complete! Transcribed: {processed_count} | Skipped (Already existed): {skipped_count}")

if __name__ == "__main__":
    main()