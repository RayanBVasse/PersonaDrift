# Make_humor_tables.py
# Computes auxiliary Humor Signaling rates (per 1,000 tokens) from raw chat exports.
# Output:
#   TableS_Humor_environments.csv
#   TableS_Humor_dyads.csv

from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ----------------------------
# CONFIG: update filenames here
# ----------------------------

# Map environment name -> chat export filename
ENV_FILES: Dict[str, str] = {
    "baseline": "Subj_533.txt",          # change if needed
    "tribe": "Combined_banter.txt",                    # change if needed
    "four_guys": "Anonymized_4_guys.txt",       # change if needed
    "acquaintances": "Chat_with_Raver.txt",     # change if needed
}

# Map dyad label -> chat export filename (if you have raw dyad .txt exports)
DYAD_FILES: Dict[str, str] = {
    "WZP_F": "WZP_F.txt",   # change if needed or delete if not available
    "WZP_K": "WZP_K.txt",
    "WZP_S": "WZP_S.txt",
    "WZP_T": "WZP_T.txt",
}

# Subject author labels as they appear in each file (exact match after cleaning)
# Add aliases if needed.
SUBJECT_AUTHOR_BY_ENV: Dict[str, List[str]] = {
    "baseline": ["[Subj_533]", "Subj_533"],
    "tribe": ["Subj_533"],
    "four_guys": ["🔆 Mr_Y", "Mr_Y"],
    "acquaintances": ["Subj_533"],
}

SUBJECT_AUTHOR_BY_DYAD: Dict[str, List[str]] = {
    "WZP_F": ["Subj_533"],
    "WZP_K": ["Subj_533"],
    "WZP_S": ["Subj_533"],
    "WZP_T": ["Subj_533"],
}

# Humor markers: overt signaling only
HUMOR_TOKENS = [
    "lol", "lmao", "rofl", "jk", "haha", "hahaha", "hehe", "hehehe",
]

HUMOR_EMOJI = [
    "😂", "🤣", "😆", "😄", "😅", "😉",
]

# WhatsApp-ish line pattern:
# Example: "09/04/2022, 13:05 - Name: message..."
# Adjust if your export format differs.
LINE_RE = re.compile( r"""^(?:
        (?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s+(?P<time>\d{1,2}:\d{2})\s+-\s+(?P<speaker1>[^:]+):\s+(?P<message1>.*)
        |
        \[(?P<speaker2>.+?)\]\s*:?\s*(?P<message2>.*)
    )$""",
    re.VERBOSE,
)

# Simple tokenizer: words + emoji kept separate
WORD_RE = re.compile(r"[A-Za-z']+")


def normalize_speaker(s: str) -> str:
    return s.strip()


def count_tokens(text: str) -> int:
    # Token count approximated as word tokens (consistent, reproducible)
    return len(WORD_RE.findall(text))


def count_humor_markers(text: str) -> int:
    t = text.lower()
    count = 0

    # Count laughter tokens as whole words to avoid matching inside other words
    for tok in HUMOR_TOKENS:
        count += len(re.findall(rf"\b{re.escape(tok)}\b", t))

    # Count emoji occurrences
    for e in HUMOR_EMOJI:
        count += t.count(e)

    return count


def parse_chat_file(path: Path) -> List[Tuple[str, str, str]]:
    """
    Returns list of (speaker, message, raw_line) for matched chat lines.
    """
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue

        if m.group("speaker1") is not None:
            speaker = normalize_speaker(m.group("speaker1"))
            message = m.group("message1").strip()
        else:
            speaker = normalize_speaker(m.group("speaker2"))
            message = m.group("message2").strip()

        rows.append((speaker, message, raw))

    return rows


def aggregate_humor(label: str,path: Path,subject_aliases: List[str],) -> Dict[str, object]:
    rows = parse_chat_file(path)

    aliases_norm = set(a.strip() for a in subject_aliases)

    subj_msgs = [(spk, msg) for (spk, msg, _) in rows if normalize_speaker(spk) in aliases_norm]

    if not subj_msgs:
        # Provide helpful diagnostics
        speakers = {}
        for (spk, _, _) in rows:
            spk2 = normalize_speaker(spk)
            speakers[spk2] = speakers.get(spk2, 0) + 1
        top = sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:10]
        raise ValueError(
            f"No messages matched subject aliases for {label} in {path.name}.\n"
            f"Top speakers detected: {top}\n"
            f"Update SUBJECT_AUTHOR_* aliases in the script to match exactly."
        )

    total_tokens = 0
    humor_count = 0

    for _, msg in subj_msgs:
        total_tokens += count_tokens(msg)
        humor_count += count_humor_markers(msg)

    humor_per_1k = (humor_count / total_tokens * 1000.0) if total_tokens else 0.0

    print(
        f"[{label}] file={path.name} | subj_msgs={len(subj_msgs)} | tokens={total_tokens} | humor_markers={humor_count} | humor_per_1k={humor_per_1k:.4f}"
    )
    return {
        "label": label,
        "file": path.name,
        "tokens": total_tokens,
        "humor_markers": humor_count,
        "humor_per_1k": round(humor_per_1k, 4),
        "n_messages": len(subj_msgs),
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)



def main() -> None:
    here = Path(__file__).resolve().parent

    env_out = []
    for env, fname in ENV_FILES.items():
        p = here / fname
        if not p.exists():
            continue
        aliases = SUBJECT_AUTHOR_BY_ENV.get(env, ["Subj_533"])
        env_out.append(aggregate_humor(env, p, aliases))

    dyad_out = []
    for dyad, fname in DYAD_FILES.items():
        p = here / fname
        if not p.exists():
            continue
        aliases = SUBJECT_AUTHOR_BY_DYAD.get(dyad, ["Subj_533"])
        dyad_out.append(aggregate_humor(dyad, p, aliases))

    summary_rows = []

    for r in env_out:
        summary_rows.append({
            "group": "environment",
            "label": r["label"],
            "file": r["file"],
            "n_messages": r["n_messages"],
            "tokens": r["tokens"],
            "humor_markers": r["humor_markers"],
            "humor_per_1k": r["humor_per_1k"],
        })

    for r in dyad_out:
        summary_rows.append({
            "group": "dyad",
            "label": r["label"],
            "file": r["file"],
            "n_messages": r["n_messages"],
            "tokens": r["tokens"],
            "humor_markers": r["humor_markers"],
            "humor_per_1k": r["humor_per_1k"],
        })

    # write summary
    with (here / "Humor_run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote {len(summary_rows)} rows to Humor_run_summary.csv")

    # Save
    env_fields = ["label", "file", "n_messages", "tokens", "humor_markers", "humor_per_1k"]
    dyad_fields = ["label", "file", "n_messages", "tokens", "humor_markers", "humor_per_1k"]

    write_csv(here / "TableS_Humor_environments.csv", env_out, env_fields)
    write_csv(here / "TableS_Humor_dyads.csv", dyad_out, dyad_fields)

    print(f"Wrote {len(env_out)} rows to TableS_Humor_environments.csv")
    print(f"Wrote {len(dyad_out)} rows to TableS_Humor_dyads.csv")


if __name__ == "__main__":
    main()
