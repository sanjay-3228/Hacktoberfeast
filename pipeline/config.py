import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

JOBS_DIR = Path(os.getenv("JOBS_DIR", "jobs"))

# Whisper: "large-v3" on GPU, "small" if you only have CPU
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")

# pyannote: newer releases may use a different model name; check the
# Hugging Face page and accept its terms first.
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
HF_TOKEN = os.getenv("HF_TOKEN", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM (any OpenAI-compatible endpoint, or Azure OpenAI)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Tuning knobs (adjust on your own test clips)
MERGE_GAP_SEC = 1.0          # join same-speaker messages closer than this
FACE_FPS = 5                 # frames per second sampled for faces
ON_CAMERA_THRESHOLD = 0.2    # min correlation to call a speaker "on camera"
MIN_TRACK_SAMPLES = 8        # ignore face tracks shorter than this
TRACK_MAX_GAP_SEC = 2.0      # close a face track after this long unseen
CHUNK_SECONDS = 45           # target length of search chunks
