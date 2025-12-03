# Modern Audio Processing Guide

This guide covers deploying state-of-the-art audio models on Chutes, specifically focusing on **Kokoro** for high-quality Text-to-Speech (TTS) and **Whisper v3** for Speech-to-Text (STT) transcription.

## High-Quality TTS with Kokoro-82M

Kokoro is a frontier TTS model that produces extremely natural-sounding speech despite its small size (82M parameters).

### 1. Define the Image

Kokoro requires specific system dependencies (`espeak-ng`, `git-lfs`) and Python packages (`phonemizer`, `scipy`, etc.).

```python
from chutes.image import Image

image = (
    Image(
        username="myuser",
        name="kokoro-82m",
        tag="0.0.1",
        readme="## Text-to-speech using hexgrade/Kokoro-82M",
    )
    .from_base("parachutes/base-python:3.12.7")
    # Install system dependencies as root
    .set_user("root")
    .run_command("apt update && apt install -y espeak-ng git-lfs")
    # Switch back to chutes user for python packages
    .set_user("chutes")
    .run_command("pip install phonemizer scipy munch torch transformers")
    # Download model weights into the image
    .run_command("git lfs install")
    .run_command("git clone https://huggingface.co/hexgrad/Kokoro-82M")
    .run_command("mv -f Kokoro-82M/* /app/")
)
```

### 2. Define the Chute & Schemas

```python
from enum import Enum
from io import BytesIO
import uuid
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from chutes.chute import Chute, NodeSelector

class VoicePack(str, Enum):
    DEFAULT = "af"
    BELLA = "af_bella"
    SARAH = "af_sarah"
    ADAM = "am_adam"
    MICHAEL = "am_michael"

class InputArgs(BaseModel):
    text: str
    voice: VoicePack = Field(default=VoicePack.DEFAULT)

chute = Chute(
    username="myuser",
    name="kokoro-tts",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24),
)
```

### 3. Initialize & Define Endpoint

We load the model and voice packs into GPU memory on startup for low-latency inference.

```python
@chute.on_startup()
async def initialize(self):
    from models import build_model
    import torch
    import wave
    import numpy as np
    from kokoro import generate

    # Store libraries for use in the endpoint
    self.wave = wave
    self.np = np
    self.generate = generate

    # Load model
    self.model = build_model("kokoro-v0_19.pth", "cuda")

    # Pre-load voice packs
    self.voice_packs = {}
    for voice_id in VoicePack:
        self.voice_packs[voice_id.value] = torch.load(
            f"voices/{voice_id.value}.pt", weights_only=True
        ).to("cuda")

@chute.cord(
    public_api_path="/speak",
    method="POST",
    output_content_type="audio/wav"
)
async def speak(self, args: InputArgs) -> StreamingResponse:
    # Generate audio
    audio_data, _ = self.generate(
        self.model, 
        args.text, 
        self.voice_packs[args.voice.value], 
        lang=args.voice.value[0]
    )
    
    # Convert to WAV
    buffer = BytesIO()
    audio_int16 = (audio_data * 32768).astype(self.np.int16)
    with self.wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_int16.tobytes())
    
    buffer.seek(0)
    return StreamingResponse(
        buffer, 
        media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={uuid.uuid4()}.wav"}
    )
```

## Speech Transcription with Whisper v3

Deploying OpenAI's Whisper Large v3 allows for state-of-the-art transcription and translation.

### 1. Setup

```python
from chutes.image import Image
from chutes.chute import Chute, NodeSelector
from pydantic import BaseModel, Field
import tempfile
import base64

# Simple image with transformers and acceleration
image = (
    Image(username="myuser", name="whisper-v3", tag="1.0")
    .from_base("parachutes/base-python:3.12.7")
    .run_command("pip install transformers torch accelerate")
)

chute = Chute(
    username="myuser",
    name="whisper-v3",
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=24)
)

class TranscriptionArgs(BaseModel):
    audio_b64: str = Field(..., description="Base64 encoded audio file")
    language: str = Field(None, description="Target language code (e.g., 'en', 'fr')")
```

### 2. Initialize Pipeline

```python
@chute.on_startup()
async def load_model(self):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import torch

    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    ).to("cuda")
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    self.pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda",
    )
```

### 3. Transcription Endpoint

```python
@chute.cord(public_api_path="/transcribe", method="POST")
async def transcribe(self, args: TranscriptionArgs):
    # Decode base64 audio to temporary file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav") as tmpfile:
        tmpfile.write(base64.b64decode(args.audio_b64))
        tmpfile.flush()
        
        generate_kwargs = {}
        if args.language:
            generate_kwargs["language"] = args.language
            
        result = self.pipe(
            tmpfile.name, 
            return_timestamps=True, 
            generate_kwargs=generate_kwargs
        )
        
        # Format chunks for cleaner output
        formatted_chunks = [
            {
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1],
                "text": chunk["text"]
            }
            for chunk in result["chunks"]
        ]
        
        return {"text": result["text"], "chunks": formatted_chunks}
```

## Usage Tips

1.  **Latency**: For real-time applications (like voice bots), prefer smaller models or streaming architectures. Kokoro is extremely fast and suitable for near real-time use.
2.  **Audio Format**: When sending audio to the API, standard formats like WAV or MP3 are supported. For base64 uploads, ensure you strip any data URI headers (e.g., `data:audio/wav;base64,`) before sending.
3.  **VRAM**: `whisper-large-v3` typically requires ~10GB VRAM for inference. Kokoro is very lightweight (<4GB). A single 24GB GPU (e.g., A10G, 3090, 4090) can easily host both if combined into one chute!

