import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# CONFIG (frozen)
# -------------------------
FILES = {
    "baseline": "Subj_533.txt",
    "tribe": "Combined_banter.txt",
    "four_guys": "Anonymized_4_guys.txt",
    "acquaintances": "Chat_with_Raver.txt",
}

SUBJECT_CANDIDATES = ["Subj_533", "Mr_Y", "Me"]

WA_RE = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2})(?:\s?[APMapm]{0,2})?\s-\s([^:]+?):\s(.*)$'
)

EMOTION_WORDS = set([
    "feel","feeling","felt","emotion","emotional","sad","happy","angry","mad","furious","afraid","scared",
    "anxious","anxiety","fear","joy","love","hurt","pain","lonely","depressed","depression","tired","exhausted",
    "stressed","stress","calm","peace","grateful","thankful","ashamed","guilt","guilty","regret","hope","hoping",
    "excited","frustrated","upset","cry","crying","tear","tears","smile","laugh","laughing"
])

REPAIR = [
    r"\bsorry\b", r"\bapolog", r"\bmy bad\b", r"\bi was wrong\b", r"\bforgive\b", r"\blet'?s fix\b",
    r"\bresolve\b", r"\bmake it right\b", r"\bi understand\b", r"\bi hear you\b", r"\bmy fault\b"
]

HEDGE = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bi think\b", r"\bi guess\b", r"\bnot sure\b", r"\bprobably\b",
    r"\bkinda\b", r"\bsort of\b", r"\bseems\b", r"\bpossibly\b"
]

COORD = [
    r"\btomorrow\b", r"\btoday\b", r"\btonight\b", r"\bthis week\b", r"\bnext week\b", r"\bwhen\b", r"\bwhere\b",
    r"\btime\b", r"\bmeet\b", r"\bmeeting\b", r"\bcall\b", r"\btext\b", r"\bplan\b", r"\bschedule\b", r"\bbook\b",
    r"\breserve\b", r"\btrain\b", r"\bgym\b", r"\bflight\b", r"\baddress\b"
]

ABSTRACT = [
    r"\bmeaning\b", r"\bpurpose\b", r"\bphilosophy\b", r"\btheory\b", r"\bframework\b", r"\bidentity\b",
    r"\bpsycholog", r"\bneurosci", r"\bconcept\b", r"\bmodel\b", r"\bmeta\b", r"\bexistential\b"
]

FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)

AXIS_MAP = {
    "Emotional expression": "emotion_words_per1k",
    "Emotional labor": "repair_markers_per1k",
    "Role constraint": "hedge_markers_per1k",
    "Identity continuity": "first_person_per1k",
    "Cognitive style": "abstract_markers_per1k",
    "Social obligation": "coord_markers_per1k",
}

# -------------------------
# HELPERS
# -------------------------
def token_words(text: str):
    return re.findall(r"\b\w+\b", text.lower())

def parse_whatsapp_txt(fp: str) -> pd.DataFrame:
    rows = []
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = WA_RE.match(line)
            if m:
                date_s, time_s, speaker, text = m.group(1), m.group(2), m.group(3).strip(), m.group(4)
                parts = date_s.split("/")
                # normalize 2-digit years
                if len(parts[-1]) == 2:
                    yy = int(parts[-1])
                    year = yy + (2000 if yy < 70 else 1900)
                    date_s = f"{int(parts[0]):02d}/{int(parts[1]):02d}/{year:04d}"
                else:
                    date_s = f"{int(parts[0]):02d}/{int(parts[1]):02d}/{int(parts[2]):04d}"

                dt = None
                for fmt in ("%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"):
                    try:
                        dt = datetime.strptime(date_s + " " + time_s, fmt)
                        break
                    except ValueError:
                        pass
                rows.append([dt, speaker, text])
            else:
                if rows and line and not line.startswith("Messages and calls are end-to-end encrypted"):
                    rows[-1][2] += "\n" + line

    df = pd.DataFrame(rows, columns=["datetime", "speaker", "text"])
    if len(df) == 0:
        # fallback: plain lines
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        df = pd.DataFrame({"datetime": pd.NaT, "speaker": "Subj_533", "text": lines})
    return df

def infer_subject_speaker(df: pd.DataFrame) -> str:
    speakers = set(df["speaker"].astype(str).unique())
    for c in SUBJECT_CANDIDATES:
        if c in speakers:
            return c
    if len(speakers) == 1:
        return list(speakers)[0]
    return df["speaker"].value_counts().idxmax()

def filter_subject_messages(df: pd.DataFrame, subject_label: str) -> pd.DataFrame:
    if subject_label in df["speaker"].astype(str).unique():
        out = df[df["speaker"].astype(str) == subject_label].copy()
        out["speaker"] = "Subj_533"
        return out

    # text-tag fallback
    mask = df["text"].astype(str).str.startswith("[user]")
    if mask.any():
        out = df[mask].copy()
        out["text"] = out["text"].astype(str).str.replace(r'^\[user\]\s*', '', regex=True)
        out["speaker"] = "Subj_533"
        return out

    # last resort: treat all as subject
    out = df.copy()
    out["speaker"] = "Subj_533"
    return out

def contains_count(texts: pd.Series, patterns) -> int:
    c = 0
    for pat in patterns:
        c += int(texts.str.contains(pat, regex=True, case=False, na=False).sum())
    return c

def compute_metrics(df_subj: pd.DataFrame) -> dict:
    texts = df_subj["text"].astype(str)
    words = texts.apply(token_words)
    wc = words.apply(len)
    total_words = int(wc.sum())
    msgs = len(df_subj)
    denom = max(total_words, 1)

    emo_hits = int(words.apply(lambda ws: sum(1 for w in ws if w in EMOTION_WORDS)).sum())
    fp_hits = int(texts.apply(lambda s: len(FIRST_PERSON.findall(s))).sum())

    repair_hits = contains_count(texts, REPAIR)
    hedge_hits = contains_count(texts, HEDGE)
    coord_hits = contains_count(texts, COORD)
    abstract_hits = contains_count(texts, ABSTRACT)

    per1k = lambda x: (x / denom) * 1000.0

    return {
        "messages": msgs,
        "total_words": total_words,
        "mean_words_per_msg": total_words / max(msgs, 1),
        "emotion_words_per1k": per1k(emo_hits),
        "repair_markers_per1k": per1k(repair_hits),
        "hedge_markers_per1k": per1k(hedge_hits),
        "coord_markers_per1k": per1k(coord_hits),
        "abstract_markers_per1k": per1k(abstract_hits),
        "first_person_per1k": per1k(fp_hits),
    }

def build_longitudinal(df_subj: pd.DataFrame, n_bins=14) -> pd.DataFrame:
    d = df_subj.copy().reset_index(drop=True)

    if d["datetime"].notna().any():
        d = d.dropna(subset=["datetime"]).reset_index(drop=True)
        t = d["datetime"].astype("int64")
        edges = np.linspace(t.min(), t.max(), n_bins + 1)
        d["bin"] = np.digitize(t, edges[1:-1]) + 1
    else:
        d["bin"] = pd.qcut(d.index + 1, n_bins, labels=False, duplicates="drop") + 1

    texts = d["text"].astype(str)
    words = texts.apply(token_words)
    d["wc"] = words.apply(len)
    d["emo_hits"] = words.apply(lambda ws: sum(1 for w in ws if w in EMOTION_WORDS))
    d["coord_hits"] = texts.str.contains("|".join(COORD), regex=True, case=False, na=False).astype(int)
    d["abstract_hits"] = texts.str.contains("|".join(ABSTRACT), regex=True, case=False, na=False).astype(int)

    out = d.groupby("bin").apply(lambda g: pd.Series({
        "msgs": len(g),
        "words": int(g["wc"].sum()),
        "emo_rate": (g["emo_hits"].sum() / max(g["wc"].sum(), 1)) * 1000.0,
        "coord_rate": (g["coord_hits"].sum() / max(g["wc"].sum(), 1)) * 1000.0,
        "abstract_rate": (g["abstract_hits"].sum() / max(g["wc"].sum(), 1)) * 1000.0,
    })).reset_index()

    return out

def plot_long(long_dfs, metric, ylabel, title, outpath):
    plt.figure(figsize=(8, 4.5))
    for env, ld in long_dfs.items():
        plt.plot(ld["bin"], ld[metric], marker="o", label=env)
    plt.xlabel("Timeline bin (1→14)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -------------------------
# RUN
# -------------------------
corpora = {}
for env, fp in FILES.items():
    df = parse_whatsapp_txt(fp)
    subj_label = infer_subject_speaker(df)
    corpora[env] = filter_subject_messages(df, subj_label)

# Table 1: structural stats
struct_rows = []
for env, d in corpora.items():
    wc = d["text"].astype(str).str.findall(r"\b\w+\b").apply(len)
    has_ts = bool(d["datetime"].notna().any())
    active_days = None
    if has_ts:
        start, end = d["datetime"].min(), d["datetime"].max()
        active_days = int((end - start).days) if pd.notna(start) and pd.notna(end) else None

    struct_rows.append({
        "environment": env,
        "messages": len(d),
        "total_words": int(wc.sum()),
        "mean_words/msg": float(wc.mean()),
        "median_words/msg": float(wc.median()),
        "p90_words/msg": float(wc.quantile(0.9)),
        "has_timestamps": has_ts,
        "active_days": active_days,
    })

struct_df = pd.DataFrame(struct_rows)
struct_df.to_csv("Table1_structural_stats.csv", index=False)

# Table 2/3: axis raw + scaled
raw_metrics = []
for env, d in corpora.items():
    m = compute_metrics(d)
    m["environment"] = env
    raw_metrics.append(m)

raw_df = pd.DataFrame(raw_metrics).set_index("environment")
axis_raw = raw_df[[AXIS_MAP[k] for k in AXIS_MAP]].copy()
axis_raw.columns = list(AXIS_MAP.keys())
axis_raw.reset_index().to_csv("Table2_axis_raw_per1k.csv", index=False)

axis_scaled = (axis_raw - axis_raw.min()) / (axis_raw.max() - axis_raw.min()).replace(0, np.nan)
axis_scaled = (axis_scaled * 10).fillna(5.0)
axis_scaled.reset_index().to_csv("Table3_axis_scaled_0_10.csv", index=False)

# Figure 1: radar
labels = list(AXIS_MAP.keys())
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
for env in axis_scaled.index:
    vals = axis_scaled.loc[env, labels].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, label=env)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_ylim(0, 10)
ax.set_title("Figure 1. Persona drift across environments (Subject 533; scaled 0–10)")
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
plt.tight_layout()
plt.savefig("Figure1_radar_env.png", dpi=200)
plt.close()

# Figure 2-4: longitudinal
long_dfs = {env: build_longitudinal(d, n_bins=14) for env, d in corpora.items()}

plot_long(long_dfs, "emo_rate", "Emotion-word rate per 1k words",
          "Figure 2. Longitudinal emotional expression proxy", "Figure2_long_emotion.png")

plot_long(long_dfs, "coord_rate", "Coordination marker rate per 1k words",
          "Figure 3. Longitudinal social obligation proxy", "Figure3_long_coordination.png")

plot_long(long_dfs, "abstract_rate", "Abstract marker rate per 1k words",
          "Figure 4. Longitudinal cognitive style proxy", "Figure4_long_abstract.png")

print("Done. Wrote: Table1_structural_stats.csv, Table2_axis_raw_per1k.csv, Table3_axis_scaled_0_10.csv, Figure1_radar_env.png, Figure2-4 PNGs.")
