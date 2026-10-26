"""Stage 2: speech to text with word-level timestamps (faster-whisper)."""
from pathlib import Path

import config
from utils import free_gpu, load_json, save_json


def transcribe(audio_path: Path, out_path: Path) -> dict:
    if out_path.exists():
        return load_json(out_path)

    import torch
    from faster_whisper import WhisperModel

    cuda = torch.cuda.is_available()
    model = WhisperModel(
        config.WHISPER_MODEL,
        device="cuda" if cuda else "cpu",
        compute_type="float16" if cuda else "int8",
    )
    segments, info = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=True
    )

    segs = []
    for s in segments:
        segs.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in (s.words or [])
            ],
        })
    data = {"language": info.language, "segments": segs}
    save_json(data, out_path)

    del model
    free_gpu()
    return data
