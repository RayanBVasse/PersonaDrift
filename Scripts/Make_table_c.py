"""
Make_table_c.py

Generate Supplementary Table C1 from four environment text files in the current folder.

Inputs (expected in same folder):
  - Anonymized_4_guys.txt
  - Chat_with_Raver.txt
  - Combined_banter.txt
  - Subj_533.txt

Outputs:
  - TableC1_bins.csv    : bin-level values (env x bin)
  - TableC1_summary.csv : summary values (env x metric) with mean, sd, nbins

Assumptions:
  - Subject 533 appears as:
      four_guys      -> "🔆 Mr_Y"   (anonymized group handle)
      all other envs -> "Subj_533"
  - Binning uses 14 equal message-count bins in chronological order (if timestamps exist).
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
    "four_guys": "🔆 Mr_Y",
    "acquaintances": "Subj_533",
    "baseline": "[Subj_533]",
    "tribe": "[Subj_533]",
}


# -----------------------------
# 2) Proxy lexicons (align with your longitudinal figures)
#    Keep these stable unless you explicitly re-define proxies in the manuscript.
# -----------------------------
EMO_WORDS = {
    "sad", "sadness", "happy", "happiness", "angry", "anger", "anxious", "anxiety",
    "fear", "scared", "love", "lonely", "excited", "depressed", "frustrated", "joy",
    "hurt", "cry", "tears", "upset", "stress", "stressed"
}

COORD_WORDS = {
    "should", "must", "need", "ought", "responsible", "duty", "obligation",
    "supposed", "required", "expect", "expected", "have", "have-to"
}

ABSTRACT_WORDS = {
    "think", "realize", "understand", "meaning", "because", "concept", "reflect",
    "consider", "perspective", "pattern", "assume", "belief", "idea", "theory",
    "reason", "logic", "interpret", "analysis"
}


# -----------------------------
# 3) Parsers for common chat formats
# -----------------------------

# WhatsApp export variants:
# "12/01/2024, 09:15 - Name: message"
WA_RE = re.compile(
    r"^\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),\s*"
    r"(\d{1,2}:\d{2})(?:\s*([AP]M))?\s*-\s*"
    r"([^:]+):\s*(.*)\s*$"
)

# Simple "Name: message"
SIMPLE_RE = re.compile(r"^\s*([^:]{1,80}):\s*(.+)\s*$")


def _parse_datetime(date_str: str, time_str: str, ampm: str | None) -> datetime | None:
    """Best-effort parse for WhatsApp date/time strings."""
    try:
        date_str = date_str.replace("-", "/")
        parts = date_str.split("/")
        if len(parts[-1]) == 2:
            parts[-1] = "20" + parts[-1]
            date_str = "/".join(parts)
        full_time = f"{time_str} {ampm}" if ampm else time_str

        fmts = [
            "%d/%m/%Y %H:%M",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %I:%M %p",
            "%m/%d/%Y %I:%M %p",
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
    """
    Parse a text chat file into a dataframe with columns:
      idx, timestamp (datetime or NaT), author, text
    """
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

            # Continuation line: append to previous message if possible
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
        p = p.strip(".,!?;:()[]{}\"'“”‘’")
        if p:
            toks.append(p)
    return toks


def count_hits(tokens: list[str], lex: set[str]) -> int:
    return sum(1 for t in tokens if t in lex)


# -----------------------------
# 5) Binning and per-bin rates
# -----------------------------
def assign_equal_count_bins(df: pd.DataFrame, n_bins: int = 14) -> pd.DataFrame:
    """
    Assign 14 equal message-count bins across chronological order.
    If <14 messages, bins become 1..N.
    """
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
    Given df with 'text' and 'bin', compute per-bin rates per 1k tokens for the 3 proxies.
    """
    toks = df["text"].fillna("").apply(tokenize)
    df["_words"] = toks.apply(len)
    df["_emo"] = toks.apply(lambda x: count_hits(x, EMO_WORDS))
    df["_coord"] = toks.apply(lambda x: count_hits(x, COORD_WORDS))
    df["_abs"] = toks.apply(lambda x: count_hits(x, ABSTRACT_WORDS))

    df = df[df["_words"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["bin", "msgs", "words", "emo_rate", "coord_rate", "abstract_rate"])

    out = (
        df.groupby("bin", as_index=False)
          .agg(
              msgs=("text", "size"),
              words=("_words", "sum"),
              emo=("_emo", "sum"),
              coord=("_coord", "sum"),
              abstract=("_abs", "sum"),
          )
          .sort_values("bin")
          .reset_index(drop=True)
    )

    out["emo_rate"] = 1000.0 * out["emo"] / out["words"]
    out["coord_rate"] = 1000.0 * out["coord"] / out["words"]
    out["abstract_rate"] = 1000.0 * out["abstract"] / out["words"]

    return out[["bin", "msgs", "words", "emo_rate", "coord_rate", "abstract_rate"]]


def summarize_c1(bins_df: pd.DataFrame) -> pd.DataFrame:
    """
    bins_df: environment, bin, emo_rate, coord_rate, abstract_rate
    returns env x metric: mean, sd, nbins
    """
    long = bins_df.melt(
        id_vars=["environment", "bin"],
        value_vars=["emo_rate", "coord_rate", "abstract_rate"],
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

    # Validate expected files exist
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
            raise ValueError(f"{fname} parsed to 0 rows (file empty or unparseable).")

        df_subj = df[df["author_norm"] == subj_author].copy()

        if df_subj.empty:
            top = df["author_norm"].value_counts().head(20).to_dict()
            raise ValueError(
                f"No messages matched subject author '{subj_author}' in {fname} (env={env}).\n"
                f"Top authors detected: {top}\n"
                f"Fix by updating SUBJECT_AUTHOR_BY_ENV in this script to match exactly."
            )

        # Order: chronological if timestamps exist, else file order
        if df_subj["timestamp"].notna().sum() >= 5:
            df_subj = df_subj.sort_values(["timestamp", "idx"]).reset_index(drop=True)
        else:
            df_subj = df_subj.sort_values(["idx"]).reset_index(drop=True)

        df_subj = assign_equal_count_bins(df_subj, n_bins=14)
        bin_rates = compute_bin_rates(df_subj)
        bin_rates.insert(0, "environment", env)
        all_bins.append(bin_rates)

    bins_df = pd.concat(all_bins, ignore_index=True)

    # Write outputs in the same folder
    bins_out = base_dir / "TableC1_bins.csv"
    summary_out = base_dir / "TableC1_summary.csv"
    bins_df.to_csv(bins_out, index=False)

    summary_df = summarize_c1(bins_df[["environment", "bin", "emo_rate", "coord_rate", "abstract_rate"]])
    summary_df.to_csv(summary_out, index=False)

    # Console sanity check
    print(f"Wrote: {bins_out}")
    print(f"Wrote: {summary_out}")
    print("\nBin counts per environment:")
    print(bins_df.groupby("environment")["bin"].nunique().to_string())
    print("\nRows in TableC1_summary.csv:", len(summary_df))


if __name__ == "__main__":
    main()
