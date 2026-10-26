"""Stage 4: combine transcript words + diarization into speaker-labelled messages."""
import config


def _speaker_at(t: float, turns: list[dict]) -> str | None:
    """Speaker whose turn contains t; otherwise the nearest turn."""
    best, best_dist = None, 1e9
    for turn in turns:
        if turn["start"] <= t <= turn["end"]:
            return turn["speaker"]
        dist = min(abs(t - turn["start"]), abs(t - turn["end"]))
        if dist < best_dist:
            best, best_dist = turn["speaker"], dist
    return best


def merge(transcript: dict, turns: list[dict]) -> list[dict]:
    """Returns messages: [{"speaker": "S1", "start", "end", "text"}, ...].

    Speakers are renamed S1, S2, ... by order of first appearance.
    """
    words = [w for seg in transcript["segments"] for w in seg["words"]]
    if not words:
        return []

    labelled = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        labelled.append((_speaker_at(mid, turns) or "UNKNOWN", w))

    # rename to S1, S2, ... in order of first appearance
    rename: dict[str, str] = {}
    for spk, _ in labelled:
        if spk not in rename:
            rename[spk] = f"S{len(rename) + 1}"

    # group consecutive words from the same speaker
    messages: list[dict] = []
    for spk, w in labelled:
        sid = rename[spk]
        gap_ok = messages and (w["start"] - messages[-1]["end"]) <= config.MERGE_GAP_SEC
        if messages and messages[-1]["speaker"] == sid and gap_ok:
            messages[-1]["end"] = w["end"]
            messages[-1]["text"] += w["word"]
        else:
            messages.append({"speaker": sid, "start": w["start"], "end": w["end"],
                             "text": w["word"]})

    for m in messages:
        m["text"] = m["text"].strip()
        m["start"] = round(m["start"], 2)
        m["end"] = round(m["end"], 2)
    return [m for m in messages if m["text"]]
