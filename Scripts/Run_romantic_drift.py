# Run_romantic_drift.py
# Frozen pipeline for Subject-533 romantic dyad analysis
# Identical methodology to environment drift analysis
# NO scaling, NO interpretation, NO reuse of prior outputs

import pandas as pd
import numpy as np
import glob
import os
import re

# =========================
# CONFIG (FROZEN)
# =========================

SUBJECT_ID = "Subj_533"
DATE_FORMAT = "%d/%m/%Y"
INPUT_DIR = "./romantic_txt/"      # folder containing WZP_F.txt, WZP_T.txt, etc
OUTPUT_DIR = "./romantic_outputs/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# AXIS LEXICONS (FROZEN)
# =========================

AXES = {
    "emotional_expression": [
        "feel", "felt", "feeling", "happy", "sad", "angry", "upset", "love",
        "hurt", "afraid", "anxious", "excited", "frustrated"
    ],
    "emotional_labor": [
        "sorry", "apologize", "apology", "forgive", "forgiveness",
        "repair", "fix", "make up"
    ],
    "role_constraint": [
        "maybe", "perhaps", "i think", "i guess", "not sure", "kind of",
        "sort of", "possibly"
    ],
    "identity_continuity": [
        "i", "me", "my", "mine", "myself"
    ],
    "cognitive_style": [
        "think", "believe", "understand", "realize", "reflect", "consider",
        "concept", "idea", "meaning"
    ],
    "social_obligation": [
        "should", "need to", "have to", "must", "plan", "schedule",
        "tomorrow", "next week", "meet"
    ]
}

# =========================
# HELPERS
# =========================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def count_axis(tokens, lexicon):
    return sum(1 for t in tokens for l in lexicon if l in t)

# =========================
# MAIN LOOP
# =========================

rows_structural = []
rows_axes = []

for filepath in glob.glob(os.path.join(INPUT_DIR, "*.txt")):
    fname = os.path.basename(filepath)
    relationship_id = fname.replace(".txt", "")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    for line in lines:
        # Expected WhatsApp format:
        # dd/mm/yyyy, hh:mm - Speaker: message
        try:
            date_part, rest = line.split(",", 1)
            time_part, rest = rest.split("-", 1)
            speaker, msg = rest.split(":", 1)

            records.append({
                "date": pd.to_datetime(date_part.strip(), format=DATE_FORMAT),
                "speaker": speaker.strip(),
                "text": msg.strip()
            })
        except:
            continue

    df = pd.DataFrame(records)
    df = df.sort_values("date")

    # =========================
    # STRUCTURAL STATS
    # =========================

    subj_df = df[df["speaker"] == SUBJECT_ID]

    total_msgs = len(df)
    subj_msgs = len(subj_df)
    partner_msgs = total_msgs - subj_msgs

    tokens = []
    for t in subj_df["text"]:
        tokens.extend(tokenize(t))

    total_tokens = len(tokens)
    avg_msg_len = total_tokens / subj_msgs if subj_msgs > 0 else 0

    rows_structural.append({
        "relationship": relationship_id,
        "total_messages": total_msgs,
        "subj_messages": subj_msgs,
        "partner_messages": partner_msgs,
        "subj_total_tokens": total_tokens,
        "subj_avg_tokens_per_msg": round(avg_msg_len, 2)
    })

    # =========================
    # AXIS COUNTS (RAW PER 1K)
    # =========================

    axis_counts = {}
    for axis, lexicon in AXES.items():
        axis_counts[axis] = count_axis(tokens, lexicon)

    axis_row = {
        "relationship": relationship_id,
        "total_tokens": total_tokens
    }

    for axis, count in axis_counts.items():
        axis_row[f"{axis}_per1k"] = round((count / total_tokens) * 1000, 3) if total_tokens > 0 else 0

    rows_axes.append(axis_row)

# =========================
# WRITE OUTPUTS
# =========================

df_struct = pd.DataFrame(rows_structural)
df_axes = pd.DataFrame(rows_axes)

df_struct.to_csv(os.path.join(OUTPUT_DIR, "TableR1_structural_stats.csv"), index=False)
df_axes.to_csv(os.path.join(OUTPUT_DIR, "TableR2_axis_raw_per1k.csv"), index=False)

print("Romantic drift analysis complete.")
print("Wrote: TableR1_structural_stats.csv, TableR2_axis_raw_per1k.csv")
