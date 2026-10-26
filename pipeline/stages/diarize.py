"""Stage 3: assign anonymous speakers to Whisper speech segments.

Whisper supplies speech segments and timestamps. SpeechBrain supplies a voice
embedding for each segment, and clustering groups similar voices. This keeps
audio speaker timelines independent from the video face tracks so stage 6 can
match the two timelines.
"""
from pathlib import Path

import config
from utils import free_gpu, load_json, save_json


def _cluster_embeddings(embeddings, num_speakers: int | None):
    from sklearn.cluster import AgglomerativeClustering

    count = len(embeddings)
    if count == 1:
        return [0]
    if num_speakers:
        clusters = min(max(num_speakers, 1), count)
        model = AgglomerativeClustering(n_clusters=clusters, metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.28,
            metric="cosine",
            linkage="average",
        )
    return model.fit_predict(embeddings).tolist()


def diarize(
    audio_path: Path,
    transcript: dict,
    out_path: Path,
    num_speakers: int | None = None,
) -> list[dict]:
    """Return anonymous audio turns aligned to Whisper segment timestamps."""
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("num_speakers must be at least 1")

    if out_path.exists():
        cached_turns = load_json(out_path)
        cached_count = len({turn.get("speaker") for turn in cached_turns})
        expected_count = min(num_speakers, len(cached_turns)) if num_speakers else None
        if num_speakers is None or (cached_turns and cached_count == expected_count):
            return cached_turns

    import numpy as np
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, received {sample_rate} Hz")

    segments = [s for s in transcript.get("segments", []) if s.get("words")]
    if not segments:
        save_json([], out_path)
        return []

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    embeddings = []
    usable_segments = []
    for segment in segments:
        start = max(int(float(segment["start"]) * sample_rate), 0)
        end = min(int(float(segment["end"]) * sample_rate), len(audio))
        if end - start < int(0.5 * sample_rate):
            continue
        waveform = torch.from_numpy(audio[start:end]).unsqueeze(0)
        with torch.no_grad():
            embedding = encoder.encode_batch(waveform).squeeze().cpu().numpy()
        embeddings.append(embedding / max(float(np.linalg.norm(embedding)), 1e-12))
        usable_segments.append(segment)

    labels = _cluster_embeddings(np.asarray(embeddings), num_speakers) if embeddings else []
    turns = [
        {
            "speaker": f"AUDIO_{label:02d}",
            "start": float(segment["start"]),
            "end": float(segment["end"]),
        }
        for segment, label in zip(usable_segments, labels)
    ]
    save_json(turns, out_path)

    del encoder
    free_gpu()
    return turns
