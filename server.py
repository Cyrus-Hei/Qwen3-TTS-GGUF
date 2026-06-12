import os
import sys
import time
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, PlainTextResponse

# Set GPU device if needed (matching your setup)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Ensure python can locate the qwen3_tts_gguf package
sys.path.append(os.getcwd())

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult

# Global instances
engine = None
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
    engine = TTSEngine('model-base', verbose=False)
    print("✨ Qwen3-TTS Engine loaded and ready!")

    voices_str = get_available_voices_list()
    print(f"👥 Available Voices: [{voices_str}]")

    yield

    # --- Shutdown ---
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
            # Create a rapid lightweight stream window
            stream = engine.create_stream()
            # Assigning pre-baked TTSResult profile directly skips encoding
            stream.set_voice(voice_profile)
            
            print(f"🎤 Synthesizing with voice '{request.voice}': \"{request.input[:40]}...\"")
            result = stream.clone(
                text=request.input, 
                language=request.language, 
                zero_shot=False, 
                config=cfg
            )
            stream.join()
            
            if result:
                result.save(temp_wav_path)
                print(f"[RTF: {result.rtf:.2f}]")
            else:
                raise HTTPException(status_code=500, detail="TTS synthesis returned null output.")
        except Exception as e:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

    # 5. Queue cleanup file removal from server storage after stream completes
    background_tasks.add_task(os.remove, temp_wav_path)

    # 6. Stream content response back matching standard OpenAI clients
    return FileResponse(temp_wav_path, media_type="audio/wav", filename="speech.wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)