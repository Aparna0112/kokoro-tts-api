import os
import uuid
import shutil
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
import logging

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# JWT Authentication imports with fallback
JWT_AVAILABLE = False
jwt = None

try:
    import jwt as pyjwt
    jwt = pyjwt
    JWT_AVAILABLE = True
    print("✅ PyJWT imported successfully")
except ImportError:
    try:
        from jose import jwt
        JWT_AVAILABLE = True
        print("✅ JWT imported from python-jose")
    except ImportError:
        print("❌ No JWT library available - authentication disabled")
        class DummyJWT:
            def encode(self, payload, key, algorithm="HS256"):
                return "dummy_token"
            def decode(self, token, key, algorithms=None):
                return {"sub": "test_user", "exp": 9999999999}
        jwt = DummyJWT()
        JWT_AVAILABLE = False

from passlib.context import CryptContext
import numpy as np
import soundfile as sf
import torch

# Try to import Kokoro TTS
KOKORO_AVAILABLE = False
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
    print("✅ Kokoro TTS imported successfully")
except ImportError as e:
    print(f"❌ Kokoro TTS not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
PORT = int(os.getenv("PORT", "8000"))

# Initialize FastAPI
app = FastAPI(
    title="Kokoro TTS API with JWT Authentication",
    description="High-quality Text-to-Speech API using Kokoro-82M model with JWT auth",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Storage directories
VOICES_DIR = Path("voices")
AUDIO_DIR = Path("audio")
VOICES_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# In-memory storage
users_db: Dict[str, dict] = {}
voices_db: Dict[str, dict] = {}
audio_db: Dict[str, dict] = {}

# Global Kokoro pipeline - loaded lazily
_kokoro_pipeline = None
_pipeline_loading = False

def get_kokoro_pipeline():
    """Lazy load Kokoro pipeline"""
    global _kokoro_pipeline, _pipeline_loading
    
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline
    
    if _pipeline_loading:
        raise Exception("Kokoro pipeline is loading, please wait...")
    
    if not KOKORO_AVAILABLE:
        raise Exception("Kokoro TTS not available")
    
    try:
        _pipeline_loading = True
        logger.info("🚀 Loading Kokoro TTS pipeline...")
        
        # Force garbage collection before loading
        gc.collect()
        
        # Initialize Kokoro pipeline for American English
        _kokoro_pipeline = KPipeline(lang_code='a')
        
        # Force garbage collection after loading
        gc.collect()
        
        logger.info("✅ Kokoro TTS pipeline loaded successfully")
        return _kokoro_pipeline
        
    except Exception as e:
        logger.error(f"Failed to load Kokoro pipeline: {e}")
        raise Exception(f"Kokoro pipeline loading failed: {str(e)}")
    finally:
        _pipeline_loading = False

# Kokoro TTS Engine
class KokoroTTSEngine:
    def __init__(self):
        self.available_voices = [
            "af_alloy", "af_aoede", "af_bella", "af_echo", "af_fable", "af_heart",
            "af_nova", "af_onyx", "af_shimmer", "am_adam", "am_domi", "am_fin",
            "am_liam", "am_sarah", "bf_emma", "bf_isabella", "bf_jenny", "bf_sky",
            "bm_george", "bm_lewis", "bm_william"
        ]
        
    def generate_speech(
        self, 
        text: str, 
        voice: str = "af_bella",
        speed: float = 1.0
    ) -> tuple[np.ndarray, int]:
        """Generate speech using Kokoro TTS"""
        
        try:
            # Get pipeline (loads on first use)
            pipeline = get_kokoro_pipeline()
            
            # Clean text
            text = text.strip()
            if not text:
                raise ValueError("Text cannot be empty")
            
            # Limit text length for memory efficiency
            if len(text) > 400:
                text = text[:400]
                logger.warning("Text truncated to 400 characters for optimal quality")
            
            # Validate voice
            if voice not in self.available_voices:
                logger.warning(f"Voice '{voice}' not available, using default 'af_bella'")
                voice = "af_bella"
            
            logger.info(f"Generating speech with voice '{voice}': '{text[:50]}...'")
            
            # Force garbage collection before generation
            gc.collect()
            
            # Generate speech using Kokoro pipeline
            generator = pipeline(text, voice=voice, speed=speed)
            
            # Collect all audio chunks
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(generator):
                logger.info(f"Generated chunk {i}: {gs} tokens, {ps} phonemes")
                audio_chunks.append(audio)
            
            # Combine audio chunks
            if audio_chunks:
                final_audio = np.concatenate(audio_chunks)
            else:
                raise Exception("No audio generated")
            
            # Kokoro outputs at 24kHz
            sample_rate = 24000
            
            # Force garbage collection after generation
            gc.collect()
            
            logger.info(f"✅ Speech generated: {len(final_audio)} samples at {sample_rate}Hz")
            return final_audio, sample_rate
            
        except Exception as e:
            # Clean up on error
            gc.collect()
            logger.error(f"Kokoro TTS generation failed: {e}")
            raise Exception(f"Speech generation failed: {str(e)}")

# Initialize TTS engine
tts_engine = KokoroTTSEngine()

# Pydantic models
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(...)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=400, description="Text to synthesize (max 400 chars)")
    voice: str = Field(default="af_bella", description="Voice ID to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")

# Utility functions
def create_access_token(data: dict) -> str:
    if not JWT_AVAILABLE:
        return "dummy_token"
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def create_refresh_token(data: dict) -> str:
    if not JWT_AVAILABLE:
        return "dummy_refresh_token"
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not JWT_AVAILABLE:
        return "test_user"
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def get_audio_duration(file_path: str) -> float:
    try:
        data, samplerate = sf.read(file_path)
        return len(data) / samplerate
    except:
        return 0.0

# Routes
@app.get("/")
async def root():
    return {
        "message": "Kokoro TTS API with JWT Authentication",
        "version": "1.0.0",
        "status": "running",
        "tts_engine": "Kokoro-82M by Hexgrad",
        "tts_available": KOKORO_AVAILABLE,
        "jwt_available": JWT_AVAILABLE,
        "pipeline_loaded": _kokoro_pipeline is not None,
        "memory_optimized": True,
        "features": [
            "High-quality speech synthesis",
            "Multiple voice options", 
            "JWT authentication",
            "Memory optimized for Render",
            "Apache licensed"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "tts_engine": "Kokoro-82M",
        "kokoro_available": KOKORO_AVAILABLE,
        "jwt_available": JWT_AVAILABLE,
        "pipeline_loaded": _kokoro_pipeline is not None,
        "pipeline_loading": _pipeline_loading
    }

@app.post("/auth/register")
async def register(user: UserCreate):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = pwd_context.hash(user.password)
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.utcnow().isoformat()
    }
    
    return {"message": "User registered successfully", "username": user.username}

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    stored_user = users_db[user.username]
    if not pwd_context.verify(user.password, stored_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.username})
    refresh_token = create_refresh_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@app.get("/profile")
async def get_profile(current_user: str = Depends(verify_token)):
    if current_user not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_data = users_db[current_user].copy()
    user_data.pop("hashed_password", None)
    return user_data

@app.get("/voices")
async def list_voices(current_user: str = Depends(verify_token)):
    """List all available Kokoro voices"""
    voices = []
    
    # American Female voices
    af_voices = [
        {"voice_id": "af_alloy", "name": "Alloy", "gender": "female", "accent": "american"},
        {"voice_id": "af_aoede", "name": "Aoede", "gender": "female", "accent": "american"},
        {"voice_id": "af_bella", "name": "Bella", "gender": "female", "accent": "american"},
        {"voice_id": "af_echo", "name": "Echo", "gender": "female", "accent": "american"},
        {"voice_id": "af_fable", "name": "Fable", "gender": "female", "accent": "american"},
        {"voice_id": "af_heart", "name": "Heart", "gender": "female", "accent": "american"},
        {"voice_id": "af_nova", "name": "Nova", "gender": "female", "accent": "american"},
        {"voice_id": "af_onyx", "name": "Onyx", "gender": "female", "accent": "american"},
        {"voice_id": "af_shimmer", "name": "Shimmer", "gender": "female", "accent": "american"},
    ]
    
    # American Male voices
    am_voices = [
        {"voice_id": "am_adam", "name": "Adam", "gender": "male", "accent": "american"},
        {"voice_id": "am_domi", "name": "Domi", "gender": "male", "accent": "american"},
        {"voice_id": "am_fin", "name": "Fin", "gender": "male", "accent": "american"},
        {"voice_id": "am_liam", "name": "Liam", "gender": "male", "accent": "american"},
        {"voice_id": "am_sarah", "name": "Sarah", "gender": "male", "accent": "american"},
    ]
    
    # British Female voices
    bf_voices = [
        {"voice_id": "bf_emma", "name": "Emma", "gender": "female", "accent": "british"},
        {"voice_id": "bf_isabella", "name": "Isabella", "gender": "female", "accent": "british"},
        {"voice_id": "bf_jenny", "name": "Jenny", "gender": "female", "accent": "british"},
        {"voice_id": "bf_sky", "name": "Sky", "gender": "female", "accent": "british"},
    ]
    
    # British Male voices
    bm_voices = [
        {"voice_id": "bm_george", "name": "George", "gender": "male", "accent": "british"},
        {"voice_id": "bm_lewis", "name": "Lewis", "gender": "male", "accent": "british"},
        {"voice_id": "bm_william", "name": "William", "gender": "male", "accent": "british"},
    ]
    
    all_voices = af_voices + am_voices + bf_voices + bm_voices
    
    return {
        "voices": all_voices,
        "total": len(all_voices),
        "categories": {
            "american_female": len(af_voices),
            "american_male": len(am_voices),
            "british_female": len(bf_voices),
            "british_male": len(bm_voices)
        },
        "default_voice": "af_bella",
        "engine": "Kokoro-82M"
    }

@app.post("/load-pipeline")
async def load_pipeline(current_user: str = Depends(verify_token)):
    """Manually load the Kokoro pipeline"""
    try:
        pipeline = get_kokoro_pipeline()
        return {
            "success": True,
            "message": "Kokoro pipeline loaded successfully",
            "pipeline_loaded": True
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to load pipeline: {str(e)}",
            "pipeline_loaded": False
        }

@app.post("/synthesize")
async def synthesize_speech(
    request: SynthesizeRequest,
    current_user: str = Depends(verify_token)
):
    if not KOKORO_AVAILABLE:
        raise HTTPException(status_code=500, detail="Kokoro TTS not available")
    
    try:
        logger.info(f"Synthesis request from {current_user}: '{request.text[:50]}...'")
        
        # Generate speech using Kokoro
        wav, sample_rate = tts_engine.generate_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )
        
        # Save audio file
        audio_id = f"audio_{uuid.uuid4().hex}"
        audio_path = AUDIO_DIR / f"{audio_id}.wav"
        
        sf.write(str(audio_path), wav, sample_rate)
        
        # Store metadata
        duration = len(wav) / sample_rate
        audio_data = {
            "audio_id": audio_id,
            "owner": current_user,
            "text": request.text,
            "voice": request.voice,
            "speed": request.speed,
            "file_path": str(audio_path),
            "sample_rate": sample_rate,
            "duration": duration,
            "created_at": datetime.utcnow().isoformat()
        }
        
        audio_db[audio_id] = audio_data
        
        # Clean up memory
        del wav
        gc.collect()
        
        return {
            "success": True,
            "audio_id": audio_id,
            "message": f"Speech synthesized using Kokoro voice '{request.voice}'",
            "sample_rate": sample_rate,
            "duration": duration,
            "download_url": f"/audio/{audio_id}",
            "voice_used": request.voice,
            "speed_used": request.speed
        }
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")

@app.get("/audio/{audio_id}")
async def download_audio(audio_id: str, current_user: str = Depends(verify_token)):
    if audio_id not in audio_db:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio = audio_db[audio_id]
    if audio.get("owner") != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    file_path = audio["file_path"]
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=f"kokoro_speech_{audio_id}.wav"
    )

@app.get("/audio/{audio_id}/info")
async def get_audio_info(audio_id: str, current_user: str = Depends(verify_token)):
    if audio_id not in audio_db:
        raise HTTPException(status_code=404, detail="Audio not found")
    
    audio = audio_db[audio_id]
    if audio.get("owner") != current_user:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    audio_info = audio.copy()
    audio_info.pop("file_path", None)
    return audio_info

@app.delete("/clear-cache")
async def clear_cache(current_user: str = Depends(verify_token)):
    """Clear audio cache to free memory"""
    try:
        # Clear audio files for this user
        user_audio_ids = [
            audio_id for audio_id, audio_data in audio_db.items()
            if audio_data.get("owner") == current_user
        ]
        
        for audio_id in user_audio_ids:
            audio_data = audio_db[audio_id]
            file_path = Path(audio_data["file_path"])
            if file_path.exists():
                file_path.unlink()
            del audio_db[audio_id]
        
        # Force garbage collection
        gc.collect()
        
        return {
            "success": True,
            "message": f"Cleared {len(user_audio_ids)} audio files for user {current_user}"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to clear cache: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        workers=1
    )
