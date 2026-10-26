"""Question answering with timestamped sources."""
from services import llm
from services.jobs import load_result
from services.search import search

SYSTEM = (
    "You answer questions about a video using ONLY the transcript excerpts "
    "provided. Each excerpt has a number, a time range and a speaker. "
    "Cite excerpts like [1] or [2]. If the answer is not in the excerpts, "
    "say you could not find it in the video. Keep answers short and clear."
)


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def ask(job_id: str, question: str, k: int = 4) -> dict:
    sources = search(job_id, question, k=k)
    if not sources:
        return {"answer": "No transcript is available for this video.", "sources": []}

    result = load_result(job_id)
    names = {s["id"]: s["name"] for s in result.get("speakers", [])}
    for s in sources:
        s["speaker"] = names.get(s["speaker"], s["speaker"])

    excerpts = "\n".join(
        f"[{i + 1}] ({_mmss(s['start'])}-{_mmss(s['end'])}, {s['speaker']}): {s['text']}"
        for i, s in enumerate(sources)
    )
    prompt = f"Excerpts:\n{excerpts}\n\nQuestion: {question}"
    answer = llm.chat(SYSTEM, prompt)

    if answer is None:  # no LLM configured: return the best excerpt (dev mode)
        best = sources[0]
        answer = (
            "(No LLM configured, showing the most relevant moment.) "
            f"[{_mmss(best['start'])}] {best['speaker']}: {best['text']}"
        )
    return {"answer": answer, "sources": sources}
