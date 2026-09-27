import tempfile
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pipeline import VoiceLanguagePipeline

app = FastAPI(title="Grievance System — Voice/Language Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at server startup, not per-request — this is the slow part
# (Whisper large + IndicTrans2 onto the GPU), so it only happens once.
print("Starting up — loading models, this takes a minute...")
pipeline = VoiceLanguagePipeline()
print("Server ready.")


@app.post("/process-complaint")
async def process_complaint(audio: UploadFile = File(...)):
    """
    Accepts an audio recording (from the browser mic recorder or any
    upload), runs it through Whisper -> IndicTrans2, and returns the
    fields Member 1 owns in the shared complaints schema.
    """
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        result = pipeline.process(tmp_path)
    except Exception as e:
        import traceback
        traceback.print_exc()  # prints the full error in the server terminal
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)

    return result


@app.get("/")
def health_check():
    return {"status": "Voice/Language module (GPU pipeline) is running"}


@app.post("/process-complaint")
async def process_complaint(audio: UploadFile = File(...)):
    """
    Accepts an audio recording (from the browser mic recorder or any
    upload), runs it through Whisper -> IndicTrans2, and returns the
    fields Member 1 owns in the shared complaints schema.
    """
    suffix = os.path.splitext(audio.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        result = pipeline.process(tmp_path)
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)

    return result


@app.get("/")
def health_check():
    return {"status": "Voice/Language module (GPU pipeline) is running"}