import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# -----------------------------
# Script to assess
# a) Within-dyad variance (how much Subj_533 fluctuates over time with Partner X) 
# b) Dyad coupling strength (correlation / lagged correlation between curves) 
# c) Between-dyad contrasts for the same subject (Subj_533 with F vs K vs S vs T
# -----------------------------



# -----------------------------
# CONFIG
# -----------------------------
DATASETS = {
    "F": "WZP_F_dyad_nrc_binned_M.csv",
    "K": "WZP_K_dyad_nrc_binned_M.csv",
    "S": "WZP_S_dyad_nrc_binned_M.csv",
    "T": "WZP_T_dyad_nrc_binned_M.csv",
}

SUBJECT = "Subj_533"

OUT_A = "Table_A_within_dyad_variance.csv"
OUT_B = "Table_B_dyad_coupling.csv"
OUT_C = "Table_C_between_dyad_contrasts.csv"

# -----------------------------
# HELPERS
# -----------------------------
def load_and_compute(df):
    # emotional balance
    df["balance"] = df["positive_per1k"] - df["negative_per1k"]

    # volatility = absolute change in balance
    df = df.sort_values("bin")
    df["volatility"] = df.groupby("speaker")["balance"].diff().abs()

    return df

# -----------------------------
# A) WITHIN-DYAD VARIANCE
# -----------------------------
rows_A = []
subject_means = {}

for label, fname in DATASETS.items():
    df = pd.read_csv(fname)
    df = load_and_compute(df)

    subj = df[df["speaker"] == SUBJECT]

    rows_A.append({
        "Dyad": label,
        "Balance_mean": subj["balance"].mean(),
        "Balance_variance": subj["balance"].var(ddof=1),
        "Volatility_mean": subj["volatility"].mean(),
        "Volatility_variance": subj["volatility"].var(ddof=1),
        "N_bins": subj["bin"].nunique()
    })

    subject_means[label] = {
        "balance": subj["balance"].mean(),
        "volatility": subj["volatility"].mean()
    }

table_A = pd.DataFrame(rows_A)
table_A.to_csv(OUT_A, index=False)

# -----------------------------
# B) DYAD COUPLING
# -----------------------------
rows_B = []

for label, fname in DATASETS.items():
    df = pd.read_csv(fname)
    df = load_and_compute(df)

    speakers = df["speaker"].unique()
    partner = [s for s in speakers if s != SUBJECT][0]

    s_df = df[df["speaker"] == SUBJECT].set_index("bin")
    p_df = df[df["speaker"] == partner].set_index("bin")

    joined = s_df.join(
        p_df,
        lsuffix="_subj",
        rsuffix="_partner",
        how="inner"
    )

    rows_B.append({
        "Dyad": label,
        "Balance_corr": joined["balance_subj"].corr(joined["balance_partner"]),
        "Volatility_corr": joined["volatility_subj"].corr(joined["volatility_partner"]),
        "Balance_lag1_partner_to_subj":
            joined["balance_subj"].corr(joined["balance_partner"].shift(1)),
        "Volatility_lag1_partner_to_subj":
            joined["volatility_subj"].corr(joined["volatility_partner"].shift(1)),
        "N_bins": len(joined)
    })

table_B = pd.DataFrame(rows_B)
table_B.to_csv(OUT_B, index=False)

# -----------------------------
# C) BETWEEN-DYAD CONTRASTS
# -----------------------------
rows_C = []

for d1, d2 in combinations(subject_means.keys(), 2):
    rows_C.append({
        "Dyad_1": d1,
        "Dyad_2": d2,
        "Delta_balance_mean":
            abs(subject_means[d1]["balance"] - subject_means[d2]["balance"]),
        "Delta_volatility_mean":
            abs(subject_means[d1]["volatility"] - subject_means[d2]["volatility"])
    })

table_C = pd.DataFrame(rows_C)
table_C.to_csv(OUT_C, index=False)

print("DONE.")
print("Wrote:", OUT_A, OUT_B, OUT_C)
