"""
Make_table_c_6dim.py

Generate Supplementary Table C1 from four environment text files.
Extended version: computes all 6 communication dimensions at bin level.

Original 3: emo_rate, coord_rate, abstract_rate
Added 3:    repair_rate (Emotional Labor), hedge_rate (Role Constraint/Register Shifts),
            first_person_rate (Identity Continuity/Self-Referential Consistency)

Inputs (expected in same folder):
  - Anonymized_4_guys.txt
  - Chat_with_Raver.txt
  - Combined_banter.txt
  - Subj_533.txt

Outputs:
  - TableC1_bins_6dim.csv    : bin-level values (env x bin x 6 dimensions)
  - TableC1_summary_6dim.csv : summary values (env x metric) with mean, sd, nbins
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


# -----------------------------
# 1) Hard-coded environment files and subject aliases
# -----------------------------
ENV_FILE_MAP = {
    "four_guys": "Anonymized_4_guys.txt",
    "acquaintances": "Chat_with_Raver.txt",
    "baseline": "Combined_banter.txt",
    "tribe": "Subj_533.txt",
}

SUBJECT_AUTHOR_BY_ENV = {
    "four_guys": "\U0001f506 Mr_Y",   # 🔆 Mr_Y
    "acquaintances": "Subj_533",
    "baseline": "[Subj_533]",
    "tribe": "[Subj_533]",
}


# -----------------------------
# 2) Lexicons for ALL 6 dimensions
# -----------------------------

# Dimension 1: Emotional Expression
EMO_WORDS = {
    "sad", "sadness", "happy", "happiness", "angry", "anger", "anxious", "anxiety",
    "fear", "scared", "love", "lonely", "excited", "depressed", "frustrated", "joy",
    "hurt", "cry", "tears", "upset", "stress", "stressed"
}

# Dimension 2: Emotional Labor / Repair (Self-Regulation)
REPAIR_PATTERNS = [
    r"\bsorry\b", r"\bapolog", r"\bmy bad\b", r"\bi was wrong\b", r"\bforgive\b",
    r"\blet'?s fix\b", r"\bresolve\b", r"\bmake it right\b", r"\bi understand\b",
    r"\bi hear you\b", r"\bmy fault\b"
]

# Dimension 3: Role Constraint / Hedge (Register Shifts)
HEDGE_PATTERNS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bi think\b", r"\bi guess\b", r"\bnot sure\b",
    r"\bprobably\b", r"\bkinda\b", r"\bsort of\b", r"\bseems\b", r"\bpossibly\b"
]

# Dimension 4: Cognitive Elaboration
ABSTRACT_WORDS = {
    "think", "realize", "understand", "meaning", "because", "concept", "reflect",
    "consider", "perspective", "pattern", "assume", "belief", "idea", "theory",
    "reason", "logic", "interpret", "analysis"
}

# Dimension 5: Self-Referential Consistency (Identity Continuity)
FIRST_PERSON_RE = re.compile(r"\b(i|me|my|mine|myself)\b", re.I)

# Dimension 6: Social Obligation / Coordination
COORD_WORDS = {
    "should", "must", "need", "ought", "responsible", "duty", "obligation",
    "supposed", "required", "expect", "expected", "have", "have-to"
}


# -----------------------------
# 3) Parsers
# -----------------------------
WA_RE = re.compile(
    r"^\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),\s*"
    r"(\d{1,2}:\d{2})(?:\s*([AP]M))?\s*-\s*"
    r"([^:]+):\s*(.*)\s*$"
)

SIMPLE_RE = re.compile(r"^\s*([^:]{1,80}):\s*(.+)\s*$")


def _parse_datetime(date_str: str, time_str: str, ampm: str | None) -> datetime | None:
    try:
        date_str = date_str.replace("-", "/")
        parts = date_str.split("/")
        if len(parts[-1]) == 2:
            parts[-1] = "20" + parts[-1]
            date_str = "/".join(parts)
        full_time = f"{time_str} {ampm}" if ampm else time_str

        fmts = [
            "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M",
            "%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p",
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(f"{date_str} {full_time}", fmt)
            except ValueError:
                continue
    except Exception:
        return None
    return None


def parse_txt(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = WA_RE.match(line)
            if m:
                d, t, ampm, author, msg = m.groups()
                dt = _parse_datetime(d, t, ampm)
                rows.append({"idx": i, "timestamp": dt, "author": author.strip(), "text": msg.strip()})
                continue
            m2 = SIMPLE_RE.match(line)
            if m2:
                author, msg = m2.groups()
                rows.append({"idx": i, "timestamp": None, "author": author.strip(), "text": msg.strip()})
                continue
            if rows:
                rows[-1]["text"] = (rows[-1]["text"] + " " + line.strip()).strip()
            else:
                rows.append({"idx": i, "timestamp": None, "author": "UNKNOWN", "text": line.strip()})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["author_norm"] = df["author"].astype(str).str.strip()
    return df


# -----------------------------
# 4) Tokenization + counts
# -----------------------------
def tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(r"\s+", text.strip().lower())
    toks = []
    for p in parts:
        p = p.strip('.,!?;:()[]{}"\'\u201c\u201d\u2018\u2019')
        if p:
            toks.append(p)
    return toks


def count_hits(tokens: list[str], lex: set[str]) -> int:
    return sum(1 for t in tokens if t in lex)


def count_pattern_hits(text: str, patterns: list[str]) -> int:
    """Count regex pattern matches in a single text string."""
    text_lower = text.lower()
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text_lower))
    return total


def count_first_person(text: str) -> int:
    return len(FIRST_PERSON_RE.findall(text))


# -----------------------------
# 5) Binning and per-bin rates (ALL 6 DIMENSIONS)
# -----------------------------
def assign_equal_count_bins(df: pd.DataFrame, n_bins: int = 14) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        df["bin"] = []
        return df
    if n < n_bins:
        df["bin"] = np.arange(1, n + 1)
        return df
    df["bin"] = pd.qcut(np.arange(n), q=n_bins, labels=False, duplicates="drop") + 1
    df["bin"] = df["bin"].astype(int)
    return df


def compute_bin_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given df with 'text' and 'bin', compute per-bin rates per 1k tokens
    for all 6 communication dimensions.
    """
    toks = df["text"].fillna("").apply(tokenize)
    df = df.copy()
    df["_words"] = toks.apply(len)

    # Original 3 (token-level matching)
    df["_emo"] = toks.apply(lambda x: count_hits(x, EMO_WORDS))
    df["_coord"] = toks.apply(lambda x: count_hits(x, COORD_WORDS))
    df["_abs"] = toks.apply(lambda x: count_hits(x, ABSTRACT_WORDS))

    # New 3 (pattern/regex matching on raw text)
    df["_repair"] = df["text"].fillna("").apply(lambda x: count_pattern_hits(x, REPAIR_PATTERNS))
    df["_hedge"] = df["text"].fillna("").apply(lambda x: count_pattern_hits(x, HEDGE_PATTERNS))
    df["_fp"] = df["text"].fillna("").apply(lambda x: count_first_person(x))

    df = df[df["_words"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "bin", "msgs", "words",
            "emo_rate", "coord_rate", "abstract_rate",
            "repair_rate", "hedge_rate", "first_person_rate"
        ])

    out = (
        df.groupby("bin", as_index=False)
          .agg(
              msgs=("text", "size"),
              words=("_words", "sum"),
              emo=("_emo", "sum"),
              coord=("_coord", "sum"),
              abstract=("_abs", "sum"),
              repair=("_repair", "sum"),
              hedge=("_hedge", "sum"),
              fp=("_fp", "sum"),
          )
          .sort_values("bin")
          .reset_index(drop=True)
    )

    out["emo_rate"] = 1000.0 * out["emo"] / out["words"]
    out["coord_rate"] = 1000.0 * out["coord"] / out["words"]
    out["abstract_rate"] = 1000.0 * out["abstract"] / out["words"]
    out["repair_rate"] = 1000.0 * out["repair"] / out["words"]
    out["hedge_rate"] = 1000.0 * out["hedge"] / out["words"]
    out["first_person_rate"] = 1000.0 * out["fp"] / out["words"]

    return out[["bin", "msgs", "words",
                "emo_rate", "coord_rate", "abstract_rate",
                "repair_rate", "hedge_rate", "first_person_rate"]]


def summarize_c1(bins_df: pd.DataFrame) -> pd.DataFrame:
    rate_cols = ["emo_rate", "coord_rate", "abstract_rate",
                 "repair_rate", "hedge_rate", "first_person_rate"]
    long = bins_df.melt(
        id_vars=["environment", "bin"],
        value_vars=rate_cols,
        var_name="metric",
        value_name="value"
    )
    summary = (
        long.groupby(["environment", "metric"], as_index=False)
            .agg(
                mean=("value", "mean"),
                sd=("value", "std"),
                nbins=("bin", "nunique")
            )
    )
    return summary


# -----------------------------
# 6) Main pipeline
# -----------------------------
def main() -> None:
    base_dir = Path(".").resolve()

    missing = []
    for env, fname in ENV_FILE_MAP.items():
        fp = base_dir / fname
        if not fp.exists():
            missing.append(fname)
    if missing:
        raise FileNotFoundError("Missing expected files in folder:\n" + "\n".join(missing))

    all_bins = []

    for env, fname in ENV_FILE_MAP.items():
        fp = base_dir / fname
        subj_author = SUBJECT_AUTHOR_BY_ENV[env]

        df = parse_txt(fp)
        if df.empty:
            raise ValueError(f"{fname} parsed to 0 rows.")

        df_subj = df[df["author_norm"] == subj_author].copy()

        if df_subj.empty:
            top = df["author_norm"].value_counts().head(20).to_dict()
            raise ValueError(
                f"No messages matched subject author '{subj_author}' in {fname} (env={env}).\n"
                f"Top authors detected: {top}\n"
                f"Fix by updating SUBJECT_AUTHOR_BY_ENV."
            )

        if df_subj["timestamp"].notna().sum() >= 5:
            df_subj = df_subj.sort_values(["timestamp", "idx"]).reset_index(drop=True)
        else:
            df_subj = df_subj.sort_values(["idx"]).reset_index(drop=True)

        df_subj = assign_equal_count_bins(df_subj, n_bins=14)
        bin_rates = compute_bin_rates(df_subj)
        bin_rates.insert(0, "environment", env)
        all_bins.append(bin_rates)

    bins_df = pd.concat(all_bins, ignore_index=True)

    rate_cols = ["emo_rate", "coord_rate", "abstract_rate",
                 "repair_rate", "hedge_rate", "first_person_rate"]

    bins_out = base_dir / "TableC1_bins_6dim.csv"
    summary_out = base_dir / "TableC1_summary_6dim.csv"
    bins_df.to_csv(bins_out, index=False)

    summary_df = summarize_c1(bins_df[["environment", "bin"] + rate_cols])
    summary_df.to_csv(summary_out, index=False)

    print(f"Wrote: {bins_out}")
    print(f"Wrote: {summary_out}")
    print("\nBin counts per environment:")
    print(bins_df.groupby("environment")["bin"].nunique().to_string())
    print(f"\nDimensions: {rate_cols}")
    print(f"\nRows in bins CSV: {len(bins_df)}")
    print(f"Rows in summary CSV: {len(summary_df)}")

    # Quick sanity check: print means per environment
    print("\n--- Quick check: mean rates per environment ---")
    for env in bins_df["environment"].unique():
        ed = bins_df[bins_df["environment"] == env]
        print(f"\n  {env}:")
        for col in rate_cols:
            print(f"    {col:25s}  mean={ed[col].mean():8.3f}  sd={ed[col].std():8.3f}")


if __name__ == "__main__":
    main()
