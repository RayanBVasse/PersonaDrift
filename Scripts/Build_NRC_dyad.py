import re
import pandas as pd
from datetime import datetime
from collections import Counter

# =========================
# CONFIG
# =========================
INPUT_TXT = "WZP_T.txt"
SPEAKER_A = "Subj_533"
SPEAKER_B = "Miss_T"
NRC_PATH = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
TIME_BIN = "M"   # "W" for weekly, "M" for monthly
OUTPUT_PREFIX = "WZP_T_dyad"

DATE_FMT = "%d/%m/%Y"

# =========================
# LOAD NRC LEXICON
# =========================
nrc = {}
with open(NRC_PATH, encoding="utf-8") as f:
    for line in f:
        word, emotion, val = line.strip().split("\t")
        if int(val) == 1:
            nrc.setdefault(word, set()).add(emotion)

EMOTIONS = sorted({
    e for emotions in nrc.values() for e in emotions
})

# =========================
# PARSE WHATSAPP
# =========================
pattern = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4}),\s(\d{2}:\d{2})\s-\s([^:]+):\s(.+)"
)

rows = []
with open(INPUT_TXT, encoding="utf-8") as f:
    for line in f:
        m = pattern.match(line)
        if not m:
            continue
        date, time, speaker, text = m.groups()
        if speaker not in {SPEAKER_A, SPEAKER_B}:
            continue
        ts = datetime.strptime(f"{date} {time}", f"{DATE_FMT} %H:%M")
        rows.append((ts, speaker, text.lower()))

df = pd.DataFrame(rows, columns=["timestamp", "speaker", "text"])
df.sort_values("timestamp", inplace=True)

# =========================
# NRC PER MESSAGE
# =========================
def nrc_counts(text):
    words = re.findall(r"\b[a-z]+\b", text)
    counts = Counter()
    for w in words:
        if w in nrc:
            for e in nrc[w]:
                counts[e] += 1
    return counts, len(words)

records = []
for _, r in df.iterrows():
    counts, tokens = nrc_counts(r["text"])
    rec = {
        "timestamp": r["timestamp"],
        "speaker": r["speaker"],
        "tokens": tokens
    }
    for e in EMOTIONS:
        rec[e] = counts.get(e, 0)
    records.append(rec)

msg_df = pd.DataFrame(records)

# =========================
# SAVE PER-MESSAGE OUTPUT
# =========================
msg_df.to_csv(f"{OUTPUT_PREFIX}_nrc_per_message.csv", index=False)

# =========================
# TIME-BIN AGGREGATION
# =========================
msg_df["bin"] = msg_df["timestamp"].dt.to_period(TIME_BIN)

agg = (
    msg_df
    .groupby(["bin", "speaker"])
    .sum(numeric_only=True)
    .reset_index()
)

# Per-1k normalization
for e in EMOTIONS:
    agg[f"{e}_per1k"] = agg[e] / agg["tokens"] * 1000

agg.to_csv(f"{OUTPUT_PREFIX}_nrc_binned_{TIME_BIN}.csv", index=False)

print("DONE.")
print(f"Outputs:")
print(f" - {OUTPUT_PREFIX}_nrc_per_message.csv")
print(f" - {OUTPUT_PREFIX}_nrc_binned_{TIME_BIN}.csv")
