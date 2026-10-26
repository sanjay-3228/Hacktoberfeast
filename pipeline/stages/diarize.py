"""Stage 3: who spoke when (pyannote.audio)."""
from pathlib import Path

import config
from utils import free_gpu, load_json, save_json


def diarize(audio_path: Path, out_path: Path, num_speakers: int | None = None) -> list[dict]:
    """Returns [{"speaker": "SPEAKER_00", "start": 0.5, "end": 4.2}, ...]."""
    if out_path.exists():
        return load_json(out_path)

    import torch
    from pyannote.audio import Pipeline

    try:  # newer pyannote uses `token`, older uses `use_auth_token`
        pipe = Pipeline.from_pretrained(config.DIARIZATION_MODEL, token=config.HF_TOKEN)
    except TypeError:
        pipe = Pipeline.from_pretrained(config.DIARIZATION_MODEL, use_auth_token=config.HF_TOKEN)

    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipe(str(audio_path), **kwargs)
    annotation = getattr(output, "speaker_diarization", output)  # pyannote 4 vs 3

    turns = [
        {"speaker": spk, "start": float(t.start), "end": float(t.end)}
        for t, _, spk in annotation.itertracks(yield_label=True)
    ]
    save_json(turns, out_path)

    del pipe
    free_gpu()
    return turns
