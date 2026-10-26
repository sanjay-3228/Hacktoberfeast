"""Stage 8: split messages into timestamped chunks and embed them for search."""
import config


def make_chunks(messages: list[dict]) -> list[dict]:
    chunks, cur = [], None
    for m in messages:
        if cur is None:
            cur = {"start": m["start"], "end": m["end"], "texts": [m["text"]],
                   "dur": {m["speaker"]: m["end"] - m["start"]}}
        else:
            cur["end"] = m["end"]
            cur["texts"].append(m["text"])
            cur["dur"][m["speaker"]] = cur["dur"].get(m["speaker"], 0) + (m["end"] - m["start"])
        if cur["end"] - cur["start"] >= config.CHUNK_SECONDS:
            chunks.append(cur)
            cur = None
    if cur:
        chunks.append(cur)
    return [{
        "id": i, "start": c["start"], "end": c["end"],
        "speaker": max(c["dur"], key=c["dur"].get),
        "text": " ".join(c["texts"]), "embedding": None,
    } for i, c in enumerate(chunks)]


def build_index(messages: list[dict], embed: bool = True) -> list[dict]:
    chunks = make_chunks(messages)
    if embed and chunks:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBED_MODEL)
        vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
        for c, v in zip(chunks, vecs):
            c["embedding"] = [round(float(x), 5) for x in v]
    return chunks
