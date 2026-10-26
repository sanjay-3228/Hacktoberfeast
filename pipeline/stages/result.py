"""Stage 9: assemble and validate result.json (see docs/CONTRACT.md)."""


def build_result(job_id: str, duration: float, messages: list[dict],
                 face_map: dict, analysis: dict) -> dict:
    talk: dict[str, float] = {}
    for m in messages:
        talk[m["speaker"]] = talk.get(m["speaker"], 0) + (m["end"] - m["start"])
    total = sum(talk.values()) or 1.0

    names = analysis.get("speaker_names", {}) or {}
    speakers = []
    for i, sid in enumerate(sorted(talk, key=lambda s: int(s[1:])), start=1):
        info = face_map.get(sid, {})
        speakers.append({
            "id": sid,
            "name": names.get(sid) or f"Speaker {i}",
            "on_camera": bool(info.get("on_camera", False)),
            "avatar": info.get("avatar"),
            "talk_time_pct": round(100 * talk[sid] / total),
        })

    return {
        "job_id": job_id,
        "title": analysis.get("title", ""),
        "topic": analysis.get("topic", ""),
        "duration": round(duration, 2),
        "speakers": speakers,
        "messages": messages,
        "chapters": analysis.get("chapters", []),
        "takeaways": analysis.get("takeaways", []),
    }


def validate_result(r: dict):
    """Raises AssertionError with a clear message if the contract is broken."""
    for key in ("job_id", "topic", "duration", "speakers", "messages", "chapters", "takeaways"):
        assert key in r, f"result.json missing '{key}'"
    ids = {s["id"] for s in r["speakers"]}
    for s in r["speakers"]:
        for k in ("id", "name", "on_camera", "avatar", "talk_time_pct"):
            assert k in s, f"speaker {s.get('id')} missing '{k}'"
        if not s["on_camera"]:
            assert s["avatar"] is None, "off-camera speaker must have avatar null"
    last = -1.0
    for m in r["messages"]:
        assert m["speaker"] in ids, f"message speaker {m['speaker']} not in speakers"
        assert m["start"] <= m["end"], "message start > end"
        assert m["start"] >= last - 1e-6, "messages must be sorted by start"
        last = m["start"]
    for c in r["chapters"]:
        for k in ("title", "start", "end", "summary"):
            assert k in c, f"chapter missing '{k}'"
