"""Markdown report built from result.json."""
from services.jobs import load_result


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 60:02d}:{t % 60:02d}"


def build_report(job_id: str) -> str:
    r = load_result(job_id)
    names = {s["id"]: s["name"] for s in r.get("speakers", [])}
    lines = [f"# {r.get('title') or 'Video report'}", ""]
    lines += ["## Topic", r.get("topic", ""), ""]

    lines += ["## Speakers"]
    for s in r.get("speakers", []):
        where = "on camera" if s.get("on_camera") else "off camera"
        lines.append(f"- **{s['name']}** ({where}), {s.get('talk_time_pct', 0)}% of talk time")
    lines.append("")

    lines += ["## Chapters"]
    for c in r.get("chapters", []):
        lines.append(f"- **{_mmss(c['start'])} {c['title']}**: {c.get('summary', '')}")
    lines.append("")

    lines += ["## Key takeaways"]
    lines += [f"- {t}" for t in r.get("takeaways", [])]
    lines.append("")

    lines += ["## Transcript"]
    for m in r.get("messages", []):
        lines.append(f"**[{_mmss(m['start'])}] {names.get(m['speaker'], m['speaker'])}:** {m['text']}")
        lines.append("")
    return "\n".join(lines)
