"""Stage 7: names, topic, chapters and takeaways from the transcript (LLM).

Uses the Groq SDK when GROQ_API_KEY is configured, with OpenAI-compatible and
Azure OpenAI fallbacks. See .env.example.
Keep all provider code inside `call_llm` so it is easy to swap.
"""
import json
import re

import config

SYSTEM = (
    "You analyse transcripts of videos. Reply with JSON only, no markdown, no "
    "commentary. Never invent facts that are not in the transcript."
)

PROMPT = """Transcript (format: [speaker id, start seconds] text):
{transcript}

Video duration: {duration:.0f} seconds.

Return JSON with exactly these keys:
{{
  "speaker_names": {{"S1": "name or null", "S2": "name or null"}},
  "topic": "2-3 sentences explaining what the video is about",
  "title": "short title, max 8 words",
  "chapters": [{{"title": "...", "start": 0, "end": 60, "summary": "1-2 sentences"}}],
  "takeaways": ["5 to 7 short key takeaways"]
}}

Rules:
- speaker_names: use a real name ONLY if the transcript clearly states it
  (e.g. someone is addressed or introduces themselves), otherwise null.
- chapters: 3 to 8 chapters in order, start/end in seconds, covering 0 to the
  full duration with no gaps or overlaps. Split where the topic changes.
"""


def call_llm(system: str, user: str) -> str:
    if config.GROQ_API_KEY:
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
        model = config.GROQ_MODEL
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
        )
        return resp.choices[0].message.content

    from openai import AzureOpenAI, OpenAI

    if config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY:
        client = AzureOpenAI(azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                             api_key=config.AZURE_OPENAI_API_KEY,
                             api_version=config.AZURE_OPENAI_API_VERSION)
        model = config.AZURE_OPENAI_DEPLOYMENT
    else:
        kwargs = {"api_key": config.LLM_API_KEY or "none"}
        if config.LLM_BASE_URL:
            kwargs["base_url"] = config.LLM_BASE_URL
        client, model = OpenAI(**kwargs), config.LLM_MODEL
    resp = client.chat.completions.create(
        model=model, temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return resp.choices[0].message.content


def extract_json(text: str) -> dict:
    """Parse JSON even if the model wrapped it in ``` fences or extra text."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def _fallback(messages: list[dict], duration: float) -> dict:
    return {
        "speaker_names": {}, "title": "Video", "topic": "",
        "chapters": [{"title": "Full video", "start": 0.0, "end": duration, "summary": ""}],
        "takeaways": [],
    }


def _clean_chapters(chapters: list[dict], duration: float) -> list[dict]:
    """Sort, clamp to [0, duration] and make them contiguous."""
    chapters = sorted((c for c in chapters if "start" in c), key=lambda c: c["start"])
    if not chapters:
        return [{"title": "Full video", "start": 0.0, "end": duration, "summary": ""}]
    out = []
    for i, c in enumerate(chapters):
        start = 0.0 if i == 0 else float(c["start"])
        end = float(chapters[i + 1]["start"]) if i + 1 < len(chapters) else duration
        out.append({"title": c.get("title", f"Chapter {i + 1}"), "start": round(start, 1),
                    "end": round(max(end, start), 1), "summary": c.get("summary", "")})
    return out


def analyze(messages: list[dict], duration: float) -> dict:
    """Returns {"speaker_names", "title", "topic", "chapters", "takeaways"}.

    TODO (only if you use videos longer than ~20 min): split the transcript
    into windows, get chapters per window, then ask for topic/takeaways from
    the chapter summaries.
    """
    transcript = "\n".join(f"[{m['speaker']}, {m['start']:.0f}s] {m['text']}" for m in messages)
    prompt = PROMPT.format(transcript=transcript, duration=duration)

    for attempt in range(2):
        try:
            data = extract_json(call_llm(SYSTEM, prompt))
            data["chapters"] = _clean_chapters(data.get("chapters", []), duration)
            data.setdefault("speaker_names", {})
            data.setdefault("takeaways", [])
            data.setdefault("topic", "")
            data.setdefault("title", "Video")
            return data
        except Exception as e:  # bad JSON, API error, ...
            print(f"[llm] attempt {attempt + 1} failed: {e}")
    print("[llm] using fallback (no chapters/takeaways)")
    return _fallback(messages, duration)
