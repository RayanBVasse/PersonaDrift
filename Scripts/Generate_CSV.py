import re
import csv
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG (edit these)
# -----------------------------
FILES = {
    "baseline": r".\Subj_533.txt",              # or "Combined Subject 533 threads.txt" if that's your baseline file
    "banter":   r".\Combined_banter.txt",
    "four_guys":r".\Anonymized_4_guys.txt",
    "raver":    r".\Chat_with_Raver.txt",
}

# Who is the subject in each file?
SUBJECT_SPEAKER = {
    "baseline": "[user]",       # if your baseline file uses [user] tags
    "banter":   "[ChatGPT]",       # if banter includes [user] lines as in snippets :contentReference[oaicite:4]{index=4}
    "four_guys":"🔆 Mr_Y",       # you said subject is Mr_Y in four-guys; sample shows "🔆 Mr_Y" :contentReference[oaicite:5]{index=5}
    "raver":    "🔆 Shin-Shane-Shine",  # replace if subject phone differs; sample shows this number :contentReference[oaicite:6]{index=6}
}

# Conservative segmentation thresholds
GAP_HOURS = 6
DAY_GAP_HOURS = 2
MAX_SUBJ_WORDS_PER_SEG = 2000

OUT_CSV = "segments_metrics_ctrl1.csv"

# -----------------------------
# LEXEME LISTS (transparent)
# -----------------------------
EMOTION_WORDS = set("""
sad sadness afraid fear anxious anxiety lonely hurt angry anger ashamed shame grief depressed depression
""".split())

VULN_PATTERNS = [
    r"\bi feel\b",
    r"\bi'm scared\b",
    r"\bi am scared\b",
    r"\bi'm afraid\b",
    r"\bi am afraid\b",
    r"\bit hurts\b",
    r"\bi'm struggling\b",
    r"\bi am struggling\b",
    r"\bi feel lost\b",
    r"\bi am lost\b",
]

REPAIR_PATTERNS = [
    r"\bsorry\b",
    r"\bi didn't mean\b",
    r"\bi did not mean\b",
    r"\blet me clarify\b",
    r"\bi shouldn't have\b",
    r"\bi should not have\b",
]

DIRECTIVE_WORDS = set("must should need have to gotta".split())
STANCE_MARKERS = [
    r"\bobviously\b",
    r"\bclearly\b",
    r"\bthe point is\b",
    r"\bin my view\b",
    r"\bi think\b",
    r"\bi believe\b",
]
HEDGE_MARKERS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bi guess\b",
    r"\bnot sure\b",
    r"\bkind of\b",
    r"\bsort of\b",
]

COMMITMENT_MARKERS = [
    r"\bi'll\b", r"\bi will\b", r"\blet's\b", r"\bwe need to\b", r"\bwe should\b"
]

# -----------------------------
# PARSING HELPERS
# -----------------------------
# WhatsApp-ish: "22/03/2020, 12:32 - Name: Message"
WA_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2})\s*-\s*([^:]+):\s*(.*)$")
# Alt: "[user]:" style
TAG_RE = re.compile(r"^\[(user|ChatGPT|system)\]:\s*(.*)$", re.IGNORECASE)

def parse_datetime(d, t):
    # dd/mm/yyyy
    return datetime.strptime(f"{d} {t}", "%d/%m/%Y %H:%M")

def simple_sentence_count(text: str) -> int:
    # crude but transparent
    parts = re.split(r"[.!?]+", text)
    return max(1, sum(1 for p in parts if p.strip()))

def word_list(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())

def count_regex_list(text_lower: str, patterns):
    return sum(1 for pat in patterns if re.search(pat, text_lower))

def count_all_occurrences(text_lower: str, patterns):
    # counts multiple occurrences by regex findall
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text_lower))
    return total

def is_subject_line(env, speaker):
    subj = SUBJECT_SPEAKER[env]
    return speaker.strip() == subj.strip()

def load_messages(env, path):
    msgs = []
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    current_dt = None
    current_speaker = None
    current_text = ""

    def flush():
        nonlocal current_dt, current_speaker, current_text
        if current_dt is not None and current_speaker is not None:
            msgs.append((current_dt, current_speaker, current_text.strip()))
        current_dt, current_speaker, current_text = None, None, ""

    for line in raw:
        m = WA_RE.match(line)
        if m:
            flush()
            dt = parse_datetime(m.group(1), m.group(2))
            speaker = m.group(3).strip()
            text = m.group(4).strip()
            current_dt, current_speaker, current_text = dt, speaker, text
            continue

        t = TAG_RE.match(line)
        if t:
            # Treat as pseudo-message (no timestamp); we can't segment by time here reliably.
            # We'll set dt incrementally later.
            # Use a fake timeline: each line becomes +1 minute.
            if current_dt is None:
                current_dt = datetime(2000, 1, 1, 0, 0)
                current_speaker = f"[{t.group(1)}]"
                current_text = t.group(2)
            else:
                # if speaker changes, flush
                new_speaker = f"[{t.group(1)}]"
                if new_speaker != current_speaker:
                    flush()
                    current_dt = msgs[-1][0] + timedelta(minutes=1) if msgs else datetime(2000,1,1,0,0)
                    current_speaker = new_speaker
                    current_text = t.group(2)
                else:
                    current_text += "\n" + t.group(2)
            continue

        # continuation line
        if current_dt is not None:
            current_text += "\n" + line.strip()

    flush()
    # sort by dt just in case
    msgs.sort(key=lambda x: x[0])
    return msgs

def segment_subject(env, messages):
    segs = []
    seg_id = 0

    buf_text = []
    buf_start = None
    last_dt = None
    buf_words = 0

    def close_segment(end_dt):
        nonlocal seg_id, buf_text, buf_start, last_dt, buf_words
        if buf_text:
            seg_id += 1
            segs.append({
                "segment_id": f"{env}_{seg_id:05d}",
                "environment": env,
                "speaker_id": SUBJECT_SPEAKER[env],
                "segment_start_iso": buf_start.isoformat(),
                "text": "\n".join(buf_text).strip(),
            })
        buf_text, buf_start, last_dt, buf_words = [], None, None, 0

    for dt, speaker, text in messages:
        if not is_subject_line(env, speaker):
            continue

        # boundary checks
        if last_dt is not None:
            gap = dt - last_dt
            if gap >= timedelta(hours=GAP_HOURS):
                close_segment(last_dt)
            elif dt.date() != last_dt.date() and gap >= timedelta(hours=DAY_GAP_HOURS):
                close_segment(last_dt)

        words = len(word_list(text))
        if buf_start is None:
            buf_start = dt

        if buf_words + words > MAX_SUBJ_WORDS_PER_SEG:
            close_segment(last_dt)

        buf_text.append(text)
        buf_words += words
        last_dt = dt

    close_segment(last_dt if last_dt else datetime.now())
    return segs

def compute_metrics_for_segment(seg):
    text = seg["text"]
    tl = text.lower()
    words = word_list(text)
    wc = max(1, len(words))
    sc = simple_sentence_count(text)

    # Counts
    emo = sum(1 for w in words if w in EMOTION_WORDS)
    vuln = count_all_occurrences(tl, VULN_PATTERNS)
    repair = count_all_occurrences(tl, REPAIR_PATTERNS)
    second = sum(1 for w in words if w in ("you","your","yours","u"))
    directive = sum(1 for w in words if w in DIRECTIVE_WORDS) + len(re.findall(r"\bhave to\b", tl))
    stance = count_all_occurrences(tl, STANCE_MARKERS)
    hedge = count_all_occurrences(tl, HEDGE_MARKERS)
    commit = count_all_occurrences(tl, COMMITMENT_MARKERS)

    # Normalize per 1000 words
    emo_rate = emo / wc * 1000
    vuln_rate = vuln / wc * 1000
    repair_rate = repair / wc * 1000
    directive_rate = directive / wc * 1000
    stance_rate = stance / wc * 1000
    hedge_rate = hedge / wc * 1000

    emotional_cost = 2.0*vuln_rate + 1.5*emo_rate + 1.0*repair_rate
    role_constraint = 1.8*directive_rate + 1.2*stance_rate - 0.6*hedge_rate

    return {
        "word_count": wc,
        "sentence_count": sc,
        "emotion_word_count": emo,
        "vulnerability_count": vuln,
        "repair_count": repair,
        "second_person_count": second,
        "directive_count": directive,
        "commitment_count": commit,
        "emotional_cost": emotional_cost,
        "role_constraint": role_constraint,
    }

def main():
    all_rows = []
    env_summaries = {}

    for env, fp in FILES.items():
        msgs = load_messages(env, fp)
        segs = segment_subject(env, msgs)
        rows = []
        for s in segs:
            m = compute_metrics_for_segment(s)
            row = {
                "segment_id": s["segment_id"],
                "environment": s["environment"],
                "speaker_id": s["speaker_id"],
                "segment_start_iso": s["segment_start_iso"],
                **{k: m[k] for k in [
                    "word_count","sentence_count","emotion_word_count","vulnerability_count","repair_count",
                    "second_person_count","directive_count","commitment_count"
                ]},
                "emotional_cost": m["emotional_cost"],
                "role_constraint": m["role_constraint"],
            }
            rows.append(row)
            all_rows.append(row)

        env_summaries[env] = {
            "n_segments": len(rows),
            "mean_emotional_cost": mean([r["emotional_cost"] for r in rows]) if rows else 0.0,
            "mean_role_constraint": mean([r["role_constraint"] for r in rows]) if rows else 0.0,
        }

    # Write CSV
    fieldnames = [
        "segment_id","environment","speaker_id","word_count","sentence_count",
        "emotion_word_count","vulnerability_count","repair_count",
        "second_person_count","directive_count","commitment_count","segment_start_iso",
        "emotional_cost","role_constraint"
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Print summary table
    print("\nENVIRONMENT SUMMARY (means):")
    for env, s in env_summaries.items():
        print(env, s)

    # Quick plots
    envs = list(env_summaries.keys())
    emo_vals = [env_summaries[e]["mean_emotional_cost"] for e in envs]
    role_vals = [env_summaries[e]["mean_role_constraint"] for e in envs]

    plt.figure()
    plt.bar(envs, emo_vals)
    plt.title("Mean Emotional Cost by Environment (transparent)")
    plt.ylabel("Emotional Cost (composite)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(envs, role_vals)
    plt.title("Mean Role Constraint by Environment (transparent)")
    plt.ylabel("Role Constraint (composite)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
