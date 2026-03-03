"""
Temporal Decomposition for PersonaDrift
========================================
For each environment × dimension:
  1. Fit a linear trend (OLS: score ~ bin_number)
  2. Report slope, p-value, R²
  3. Compute raw SD (what's currently in the manuscript)
  4. Compute detrended SD (residual after removing linear trend)
  5. Compute % of variance explained by trend vs residual
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load bin-level data
df = pd.read_csv("TableC1_bins.csv")

# Dimension labels for the paper
dim_labels = {
    "emo_rate": "Emotional Expression",
    "coord_rate": "Social Obligation",
    "abstract_rate": "Cognitive Elaboration"
}

dimensions = ["emo_rate", "coord_rate", "abstract_rate"]
environments = ["baseline", "tribe", "four_guys", "acquaintances"]

results = []

for env in environments:
    env_data = df[df["environment"] == env].sort_values("bin")

    for dim in dimensions:
        y = env_data[dim].values
        x = env_data["bin"].values  # 1-14
        n = len(y)

        # Raw SD (what's currently reported)
        raw_sd = np.std(y, ddof=1)
        raw_mean = np.mean(y)

        # Linear regression: y = a + b*x
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        # Detrended residuals
        predicted = intercept + slope * x
        residuals = y - predicted
        detrended_sd = np.std(residuals, ddof=1)  # residual SD (ddof=1 for df correction)

        # Variance decomposition
        raw_var = np.var(y, ddof=1)
        detrended_var = np.var(residuals, ddof=1)
        trend_var = raw_var - detrended_var
        pct_trend = (trend_var / raw_var * 100) if raw_var > 0 else 0
        pct_residual = 100 - pct_trend

        # Trend direction
        if p_value < 0.05:
            direction = "↑" if slope > 0 else "↓"
        else:
            direction = "—"

        results.append({
            "Environment": env.replace("_", " ").title(),
            "Dimension": dim_labels[dim],
            "Mean": round(raw_mean, 3),
            "Raw SD": round(raw_sd, 3),
            "Trend Slope (per bin)": round(slope, 4),
            "Trend Direction": direction,
            "Trend p": round(p_value, 4),
            "Trend R²": round(r_squared, 3),
            "Detrended SD": round(detrended_sd, 3),
            "% Variance (Trend)": round(pct_trend, 1),
            "% Variance (Residual)": round(pct_residual, 1),
        })

results_df = pd.DataFrame(results)

# Save full results
results_df.to_csv("temporal_decomposition_results.csv", index=False)

# Print formatted table
print("=" * 120)
print("TEMPORAL DECOMPOSITION: TREND vs RESIDUAL VARIABILITY")
print("=" * 120)

for env in environments:
    env_label = env.replace("_", " ").title()
    print(f"\n--- {env_label} ---")
    env_rows = results_df[results_df["Environment"] == env_label]
    for _, row in env_rows.iterrows():
        print(f"  {row['Dimension']:25s}  "
              f"Mean={row['Mean']:7.3f}  "
              f"Raw SD={row['Raw SD']:6.3f}  "
              f"Slope={row['Trend Slope (per bin)']:+.4f} {row['Trend Direction']}  "
              f"p={row['Trend p']:.4f}  "
              f"R²={row['Trend R²']:.3f}  "
              f"Detrended SD={row['Detrended SD']:6.3f}  "
              f"Trend%={row['% Variance (Trend)']:5.1f}  "
              f"Resid%={row['% Variance (Residual)']:5.1f}")

print("\n" + "=" * 120)

# Summary insights
print("\nKEY FINDINGS:")
print("-" * 60)

sig_trends = results_df[results_df["Trend p"] < 0.05]
if len(sig_trends) > 0:
    print(f"\nSignificant linear trends (p < .05): {len(sig_trends)} of {len(results_df)}")
    for _, row in sig_trends.iterrows():
        print(f"  • {row['Environment']} / {row['Dimension']}: "
              f"slope = {row['Trend Slope (per bin)']:+.4f}, "
              f"p = {row['Trend p']:.4f}, "
              f"R² = {row['Trend R²']:.3f} "
              f"({row['% Variance (Trend)']:.1f}% of variance)")
else:
    print("\nNo significant linear trends found (all p > .05)")

# Cases where detrending substantially changes the story
print(f"\nVariance explained by trend (range): "
      f"{results_df['% Variance (Trend)'].min():.1f}% – {results_df['% Variance (Trend)'].max():.1f}%")
print(f"Mean variance explained by trend: {results_df['% Variance (Trend)'].mean():.1f}%")
print(f"Mean variance in residual: {results_df['% Variance (Residual)'].mean():.1f}%")

# Biggest SD change after detrending
results_df["SD_reduction_%"] = ((results_df["Raw SD"] - results_df["Detrended SD"]) / results_df["Raw SD"] * 100).round(1)
biggest = results_df.loc[results_df["SD_reduction_%"].idxmax()]
print(f"\nLargest SD reduction after detrending: "
      f"{biggest['Environment']} / {biggest['Dimension']} "
      f"(Raw SD={biggest['Raw SD']:.3f} → Detrended SD={biggest['Detrended SD']:.3f}, "
      f"{biggest['SD_reduction_%']:.1f}% reduction)")
