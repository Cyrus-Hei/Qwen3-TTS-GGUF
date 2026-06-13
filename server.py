import os
import sys
import time
import shutil
import tempfile
import threading
import traceback as tb_module
import io
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, PlainTextResponse, Response

# Set GPU device if needed (matching your setup)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Ensure python can locate the qwen3_tts_gguf package
sys.path.append(os.getcwd())

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult

# Global instances
engine = None
# Cache for TTSStream objects keyed by voice name to eliminate redundant 
# LlamaContext allocation and expensive pre-decoding passes.
STREAM_CACHE = {}
MAX_CACHE_SIZE = 5
inference_lock = threading.Lock()

# Directory to manage custom voices
VOICES_DIR = os.path.join(os.getcwd(), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

ARCHIVE_DIR = os.path.join(VOICES_DIR, "processed_sources")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# In-memory RAM cache for pre-encoded speaker profiles
VOICE_CACHE = {}

SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a"]

class SpeechRequest(BaseModel):
    model: str = "model-base"
    input: str
    voice: str  # Maps directly to the voice profile name (e.g., "vivian")
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0
    temperature: Optional[float] = 0.6
    language: Optional[str] = "English" # Optional language tag override
    
def get_available_voices_list() -> str:
    """
    Scans the voices directory and returns a sorted, comma-separated 
    string of unique voice names with absolutely no trailing spaces.
    """
    if not os.path.exists(VOICES_DIR):
        return ""
    
    found_voices = set()
    all_files = os.listdir(VOICES_DIR)
    
    # Build a quick lookup set of all text transcript names
    txt_basenames = {os.path.splitext(f)[0] for f in all_files if f.endswith(".txt")}
    
    for filename in all_files:
        basename, ext = os.path.splitext(filename)
        # Condition A: It is a pre-saved lightweight JSON voice profile
        if ext == ".json":
            found_voices.add(basename)
        # Condition B: It is a raw audio file that has its accompanying text transcript
        elif ext in SUPPORTED_EXTENSIONS:
            if basename in txt_basenames:
                found_voices.add(basename)
                
    # Sort alphabetically and join strictly with a comma (no spaces)
    return ",".join(sorted(list(found_voices)))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    global engine
    print("\n🚀 Initializing Qwen3-TTS Engine...")
    # Initialize in headless mode (enable_speaker=False) for the API server
    # to prevent PortAudio conflicts with HTTP response delivery on Windows.
    engine = TTSEngine('model-base', verbose=False, enable_speaker=False)
    print("✨ Qwen3-TTS Engine loaded and ready!")

    voices_str = get_available_voices_list()
    print(f"👥 Available Voices: [{voices_str}]")

    yield

    # --- Shutdown ---
    if STREAM_CACHE:
        print(f"\n🛑 Shutting down {len(STREAM_CACHE)} cached streams...")
        for s in STREAM_CACHE.values():
            s.shutdown()
        STREAM_CACHE.clear()
    if engine:
        print("\n🛑 Shutting down TTS engine...")
        engine.shutdown()

app = FastAPI(title="Qwen3-TTS OpenAI-Compatible API Server", lifespan=lifespan)

@app.get("/v1/audio/speech/speakers_list", response_class=PlainTextResponse)
def list_voices():
    """
    Endpoint that returns a strict comma-separated list of 
    available voices without any trailing spaces.
    """
    return get_available_voices_list()

def get_or_create_voice_profile(voice_name: str) -> TTSResult:
    """
    Retrieves a cached voice profile object from RAM, loads from JSON, 
    or dynamically finds and encodes a matching audio file.
    """
    if voice_name in VOICE_CACHE:
        return VOICE_CACHE[voice_name]

    json_path = os.path.join(VOICES_DIR, f"{voice_name}.json")
    txt_path = os.path.join(VOICES_DIR, f"{voice_name}.txt")

    # --- TRACK DISK READ TIMING ---
    if os.path.exists(json_path):
        print(f"📦 Loading pre-encoded voice profile for '{voice_name}' from JSON...")
        start_read = time.time()
        
        voice_obj = TTSResult.from_json(json_path)
        
        read_duration = time.time() - start_read
        print(f"⏱️ [Profile Read Time] Loaded JSON for '{voice_name}' in {read_duration:.4f} seconds.")
        
        VOICE_CACHE[voice_name] = voice_obj
        return voice_obj

    audio_path = None
    for ext in SUPPORTED_EXTENSIONS:
        potential_path = os.path.join(VOICES_DIR, f"{voice_name}{ext}")
        if os.path.exists(potential_path):
            audio_path = potential_path
            break

    # --- TRACK EMBEDDING ENCODE TIMING ---
    if audio_path:
        if not os.path.exists(txt_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Voice '{voice_name}' has an audio file but is missing its matching transcript '{voice_name}.txt'."
            )
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        print(f"🎙️ [Encoding] Processing raw audio for '{voice_name}'...")
        temp_stream = engine.create_stream()
        try:
            start_encode = time.time()
            
            temp_stream.set_voice(audio_path, ref_text)
            temp_stream.voice.save(json_path)
            
            encode_duration = time.time() - start_encode
            print(f"⏱️ [Profile Encode Time] Encoded and cached '{voice_name}' in {encode_duration:.4f} seconds.")
            
            try:
                shutil.move(audio_path, os.path.join(ARCHIVE_DIR, os.path.basename(audio_path)))
                shutil.move(txt_path, os.path.join(ARCHIVE_DIR, os.path.basename(txt_path)))
                print(f"🧹 Cleaned up: Moved raw source files for '{voice_name}' to 'voices/processed_sources/'")
            except Exception as move_err:
                print(f"⚠️ Warning: Profile saved, but could not clean up/move original source files: {move_err}")
            
            voice_obj = TTSResult.from_json(json_path)
            VOICE_CACHE[voice_name] = voice_obj
            return voice_obj
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract voice embedding: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"Voice profile '{voice_name}' not found.")

@app.post("/v1/audio/speech")
def text_to_speech(request: SpeechRequest, raw_request: Request, background_tasks: BackgroundTasks):
    # --- 🔍 DEBUG LOG BLOCK ---
    # print("\n" + "="*60)
    # print("📥 INCOMING REQUEST DEBUG LOG")
    # print("="*60)
    # print(f"Method: {raw_request.method}")
    # print(f"URL:    {raw_request.url}")
    # print(f"Client: {raw_request.client.host if raw_request.client else 'Unknown'}")
    # print(raw_request)

    # print("\n[HTTP Headers]")
    # for key, value in raw_request.headers.items():
    #     print(f"  {key}: {value}")
    
    payload_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    print("Request: ",payload_dict)  
    # print("\n[Parsed Pydantic Payload Body]")
    # Handle Pydantic v1 vs v2 compatibility gracefully
    # for key, value in payload_dict.items():
    #     print(f"  {key}: {value}")
    # print("="*60 + "\n")
    # --- 🔍 END DEBUG LOG BLOCK ---
    
    if not engine:
        raise HTTPException(status_code=500, detail="TTS Engine is not active.")

    # 1. Fetch the zero-overhead voice profile
    voice_profile = get_or_create_voice_profile(request.voice)

    # 2. Prepare generation hyper-parameters
    cfg = TTSConfig(
        max_steps=1000,
        temperature=request.temperature, 
        sub_temperature=request.temperature, 
        seed=42, 
        sub_seed=45,
        streaming=False
    )

    # 3. Acquire a localized temporary file to hold synthesized audio
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav_path = temp_wav.name
    temp_wav.close()

    # 4. Serialize inference through execution lock
    with inference_lock:
        try:
            voice_name = request.voice
            
            # Voice-Aware Stream Caching:
            # If a stream for this voice already exists, reuse it to skip 
            # the expensive pre-decoding pass of the reference audio.
            if voice_name in STREAM_CACHE:
                print(f"[DEBUG] Reusing cached stream for voice: {voice_name}")
                stream = STREAM_CACHE[voice_name]
            else:
                print(f"[DEBUG] Creating new cached stream for voice: {voice_name}")
                stream = engine.create_stream()
                # Initialize the stream with the voice profile (performs pre-decoding once)
                stream.set_voice(voice_profile)
                
                # Manage cache size to prevent OOM (LRU-ish)
                if len(STREAM_CACHE) >= MAX_CACHE_SIZE:
                    # Remove the first item (oldest)
                    oldest_voice = next(iter(STREAM_CACHE))
                    old_stream = STREAM_CACHE.pop(oldest_voice)
                    old_stream.shutdown()
                    print(f"[DEBUG] Evicted voice '{oldest_voice}' from cache.")
                
                STREAM_CACHE[voice_name] = stream

            print(f"🎤 Synthesizing with voice '{voice_name}': \"{request.input[:40]}...\"")
            print(f"[DEBUG] Calling stream.clone()...")
            # Note: stream.clone() internally calls talker.clear_memory() to reset 
            # synthesis state without destroying the voice anchor.
            result = stream.clone(
                text=request.input, 
                language=request.language, 
                zero_shot=False, 
                config=cfg
            )
            print(f"[DEBUG] stream.clone() returned, result={result is not None}")
            print(f"[DEBUG] Calling stream.join()...")
            stream.join()
            print(f"[DEBUG] stream.join() completed")
            
            if result:
                print(f"[DEBUG] Saving result to {temp_wav_path}...")
                result.save(temp_wav_path)
                print(f"[DEBUG] Result saved, printing RTF...")
                print(f"[RTF: {result.rtf:.2f}]")
                print(f"[DEBUG] RTF printed")
                
                # Speaker is disabled in headless mode for the API server, 
                # so no restart is needed here.
            else:
                raise HTTPException(status_code=500, detail="TTS synthesis returned null output.")
        except Exception as e:
            print(f"[ERROR] Synthesis failed: {e}")
            tb_module.print_exc()
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

    # 6. Read the entire WAV file into memory BEFORE any streaming begins.
    # The crash happens during uvicorn's response streaming when iterfile tries
    # to read from disk while the decoder/speaker subprocesses are still running.
    # By loading into memory first, we eliminate all file I/O during streaming,
    # which prevents the C-level crash caused by async I/O + subprocess interaction.
    print(f"[DEBUG] Reading file into memory: {temp_wav_path}")
    with open(temp_wav_path, "rb") as f:
        file_content = f.read()
    print(f"[DEBUG] File loaded into memory: {len(file_content)} bytes")
    
    # 7. Clean up temp file immediately (no longer needed on disk)
    temp_file_deleted = False
    try:
        os.remove(temp_wav_path)
        temp_file_deleted = True
        print(f"[DEBUG] Temp file deleted: {temp_wav_path}")
    except Exception as e:
        print(f"[DEBUG] Failed to delete temp file: {e}")
    
    # 8. Return as Response (synchronous, non-streaming) to bypass StreamingResponse's async iteration
    # StreamingResponse uses an async generator which may interact poorly with running subprocesses
    # Response sends bytes directly without async iteration
    print(f"[DEBUG] Returning Response with {len(file_content)} bytes...")
    try:
        resp = Response(
            content=file_content, 
            media_type="audio/wav", 
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        print(f"[DEBUG] Response returned successfully")
        return resp
    except Exception as e:
        print(f"[ERROR] Response creation failed: {e}")
        tb_module.print_exc()
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)