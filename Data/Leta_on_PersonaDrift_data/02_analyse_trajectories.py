"""
LETA x PersonaDrift Integration - Script 2: Trajectory Analysis
================================================================
Takes per-message emotion CSV from Script 1, computes:
  - 14-bin temporal trajectories per environment
  - Cross-environment LETA profiles (radar plot)
  - Trajectory stability metrics (SD, volatility)
  - Cross-environment statistical comparisons
  - Publication-ready figures and tables

Usage:
    python 02_analyze_trajectories.py --input persona_drift_leta_emotions.csv

Output:
    - leta_binned_trajectories.csv
    - leta_environment_profiles.csv
    - leta_stability_metrics.csv
    - leta_cross_environment_comparison.csv
    - Figure_LETA_radar.png
    - Figure_LETA_trajectories.png
    - Figure_LETA_heatmap.png

Requirements: pip install pandas numpy matplotlib scipy
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from scipy import stats
from pathlib import Path


CORE_EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear',
                 'joy', 'sadness', 'surprise', 'trust']

SCORE_COLS = [f'score_{e}' for e in CORE_EMOTIONS]
PER1K_COLS = [f'per1k_{e}' for e in CORE_EMOTIONS]


# =============================================================================
# TEMPORAL BINNING
# =============================================================================

def create_temporal_bins(df, n_bins=14):
    """
    Divide messages into n equal-count chronological bins per environment.
    Uses message order (msg_index) as proxy for time.
    Returns df with 'bin' column added.
    """
    results = []
    for env in df['environment'].unique():
        env_df = df[df['environment'] == env].copy()
        env_df = env_df.sort_values('msg_index').reset_index(drop=True)

        # Equal-count bins
        env_df['bin'] = pd.qcut(
            env_df.index, q=n_bins, labels=range(n_bins), duplicates='drop'
        )
        results.append(env_df)

    return pd.concat(results, ignore_index=True)


def compute_bin_statistics(df):
    """
    Compute mean emotion scores per bin per environment.
    Returns: binned_df with columns [environment, bin, n_messages, total_tokens,
             score_anger, ..., score_trust]
    """
    grouped = df.groupby(['environment', 'bin'])

    # Aggregate
    agg_dict = {
        'msg_index': 'count',
        'total_tokens': 'sum',
    }
    for col in SCORE_COLS:
        agg_dict[col] = 'mean'
    for col in PER1K_COLS:
        agg_dict[col] = 'mean'

    binned = grouped.agg(agg_dict).reset_index()
    binned.rename(columns={'msg_index': 'n_messages'}, inplace=True)

    return binned


# =============================================================================
# ENVIRONMENT PROFILES
# =============================================================================

def compute_environment_profiles(df):
    """
    Compute mean emotion profile per environment (across all messages).
    Returns: profile_df with one row per environment.
    """
    profiles = df.groupby('environment').agg({
        'msg_index': 'count',
        'total_tokens': ['sum', 'mean'],
        **{col: ['mean', 'std'] for col in SCORE_COLS},
        **{col: ['mean', 'std'] for col in PER1K_COLS},
    })

    # Flatten column names
    profiles.columns = ['_'.join(col).strip('_') for col in profiles.columns]
    profiles = profiles.reset_index()

    return profiles


# =============================================================================
# STABILITY METRICS
# =============================================================================

def compute_stability_metrics(binned_df):
    """
    Compute trajectory stability metrics per environment per emotion.
    Metrics: mean, SD across bins, CV (coefficient of variation),
             range, trend (slope of linear fit).
    """
    rows = []
    for env in binned_df['environment'].unique():
        env_data = binned_df[binned_df['environment'] == env].sort_values('bin')

        for emotion in CORE_EMOTIONS:
            col = f'per1k_{emotion}'
            values = env_data[col].values
            bins = env_data['bin'].astype(float).values

            mean_val = np.mean(values)
            sd_val = np.std(values, ddof=1) if len(values) > 1 else 0
            cv_val = sd_val / mean_val if mean_val > 0 else np.nan
            range_val = np.max(values) - np.min(values)

            # Linear trend
            if len(values) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(bins, values)
            else:
                slope, r_value, p_value = 0, 0, 1

            rows.append({
                'environment': env,
                'emotion': emotion,
                'mean_per1k': round(mean_val, 3),
                'sd_per1k': round(sd_val, 3),
                'cv': round(cv_val, 3) if not np.isnan(cv_val) else None,
                'range_per1k': round(range_val, 3),
                'trend_slope': round(slope, 4),
                'trend_r': round(r_value, 3),
                'trend_p': round(p_value, 4),
            })

    return pd.DataFrame(rows)


# =============================================================================
# CROSS-ENVIRONMENT COMPARISONS
# =============================================================================

def compute_cross_environment_tests(df):
    """
    Pairwise environment comparisons using independent t-tests and Cohen's d.
    For each emotion, compare mean per-message scores between environments.
    """
    environments = sorted(df['environment'].unique())
    rows = []

    for i, env1 in enumerate(environments):
        for env2 in environments[i+1:]:
            d1 = df[df['environment'] == env1]
            d2 = df[df['environment'] == env2]

            for emotion in CORE_EMOTIONS:
                col = f'per1k_{emotion}'
                vals1 = d1[col].values
                vals2 = d2[col].values

                # t-test (Welch's)
                t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(vals1)-1)*np.var(vals1, ddof=1) +
                     (len(vals2)-1)*np.var(vals2, ddof=1)) /
                    (len(vals1) + len(vals2) - 2)
                )
                cohens_d = (np.mean(vals1) - np.mean(vals2)) / pooled_std if pooled_std > 0 else 0

                rows.append({
                    'env_1': env1,
                    'env_2': env2,
                    'emotion': emotion,
                    'mean_1': round(np.mean(vals1), 3),
                    'mean_2': round(np.mean(vals2), 3),
                    'diff': round(np.mean(vals1) - np.mean(vals2), 3),
                    't_stat': round(t_stat, 3),
                    'p_value': round(p_val, 4),
                    'cohens_d': round(cohens_d, 3),
                    'n_1': len(vals1),
                    'n_2': len(vals2),
                })

    return pd.DataFrame(rows)


# =============================================================================
# FIGURES
# =============================================================================

def plot_radar(profiles_df, output_path='Figure_LETA_radar.png'):
    """
    Radar plot of z-scored emotion profiles across environments.
    Matches PersonaDrift Figure 1 style.
    """
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    emotions = CORE_EMOTIONS
    n_emotions = len(emotions)
    angles = np.linspace(0, 2 * np.pi, n_emotions, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    colors = {
        'baseline': '#2196F3',
        'tribe': '#4CAF50',
        'four_guys': '#FF9800',
        'acquaintances': '#9C27B0',
    }

    for _, row in profiles_df.iterrows():
        env = row['environment']
        values = [row.get(f'per1k_{e}_mean', 0) for e in emotions]

        # Z-score for visualization
        mean_all = np.mean(values)
        std_all = np.std(values)
        if std_all > 0:
            z_values = [(v - mean_all) / std_all for v in values]
        else:
            z_values = values
        z_values += z_values[:1]

        color = colors.get(env, '#666666')
        ax.plot(angles, z_values, 'o-', linewidth=2, label=env.replace('_', ' ').title(),
                color=color, markersize=5)
        ax.fill(angles, z_values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([e.capitalize() for e in emotions], size=10)
    ax.set_title('LETA Emotion Profiles Across Environments\n(z-scored per environment)',
                 size=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {output_path}")


def plot_trajectories(binned_df, output_path='Figure_LETA_trajectories.png'):
    """
    Longitudinal trajectory plots for key emotions across environments.
    Matches PersonaDrift Figures 2-4 style.
    """
    # Plot 3 key emotions: Emotional Expression (joy+trust), Fear, Sadness
    key_emotions = ['joy', 'sadness', 'fear', 'anger']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    colors = {
        'baseline': '#2196F3',
        'tribe': '#4CAF50',
        'four_guys': '#FF9800',
        'acquaintances': '#9C27B0',
    }

    for idx, emotion in enumerate(key_emotions):
        ax = axes[idx]
        col = f'per1k_{emotion}'

        for env in binned_df['environment'].unique():
            env_data = binned_df[binned_df['environment'] == env].sort_values('bin')
            color = colors.get(env, '#666666')
            ax.plot(env_data['bin'].astype(int), env_data[col],
                    'o-', label=env.replace('_', ' ').title(),
                    color=color, linewidth=2, markersize=4)

        ax.set_title(f'{emotion.capitalize()} (per 1,000 tokens)', fontweight='bold')
        ax.set_xlabel('Temporal Bin')
        ax.set_ylabel('Rate per 1,000 tokens')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('LETA Longitudinal Emotion Trajectories Across Environments',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {output_path}")


def plot_heatmap(stability_df, output_path='Figure_LETA_heatmap.png'):
    """
    Heatmap of mean emotion rates across environments.
    """
    pivot = stability_df.pivot_table(
        index='environment', columns='emotion', values='mean_per1k'
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([e.replace('_', ' ').title() for e in pivot.index])

    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > pivot.values.mean() else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    color=color, fontsize=9)

    ax.set_title('LETA Mean Emotion Rates per 1,000 Tokens', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Rate per 1,000 tokens')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Figure] Saved: {output_path}")


# =============================================================================
# MANUSCRIPT TABLE GENERATOR
# =============================================================================

def generate_manuscript_table(stability_df, comparison_df, output_path='leta_manuscript_table.txt'):
    """Generate formatted text table for manuscript insertion."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TABLE: LETA Emotion Profiles Across Environments\n")
        f.write("=" * 80 + "\n\n")

        # Table 1: Mean rates
        f.write("Mean emotion rates (per 1,000 tokens) by environment:\n\n")
        pivot = stability_df.pivot_table(
            index='emotion', columns='environment', values='mean_per1k'
        )
        f.write(pivot.to_string())
        f.write("\n\n")

        # Table 2: Stability (SD across bins)
        f.write("-" * 80 + "\n")
        f.write("Temporal variability (SD across 14 bins) by environment:\n\n")
        pivot_sd = stability_df.pivot_table(
            index='emotion', columns='environment', values='sd_per1k'
        )
        f.write(pivot_sd.to_string())
        f.write("\n\n")

        # Table 3: Key cross-environment comparisons
        f.write("-" * 80 + "\n")
        f.write("Cross-environment comparisons (Cohen's d):\n\n")
        pivot_d = comparison_df.pivot_table(
            index='emotion', columns=['env_1', 'env_2'], values='cohens_d'
        )
        f.write(pivot_d.to_string())
        f.write("\n\n")

        # Significance summary
        f.write("-" * 80 + "\n")
        f.write("Significant comparisons (p < .05):\n\n")
        sig = comparison_df[comparison_df['p_value'] < 0.05]
        for _, row in sig.iterrows():
            f.write(f"  {row['emotion']:15s} | {row['env_1']:12s} vs {row['env_2']:12s} | "
                    f"d = {row['cohens_d']:+.3f} | p = {row['p_value']:.4f}\n")

    print(f"[Table] Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LETA x PersonaDrift: Trajectory Analysis')
    parser.add_argument('--input', type=str, default='persona_drift_leta_emotions.csv',
                        help='Path to emotion CSV from Script 1')
    parser.add_argument('--bins', type=int, default=14,
                        help='Number of temporal bins (default: 14)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for results')
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} messages across {df['environment'].nunique()} environments")

    # Step 1: Temporal binning
    print(f"\n[Step 1] Creating {args.bins} temporal bins per environment...")
    df = create_temporal_bins(df, n_bins=args.bins)

    # Step 2: Compute bin statistics
    print("[Step 2] Computing bin-level statistics...")
    binned = compute_bin_statistics(df)
    binned.to_csv(outdir / 'leta_binned_trajectories.csv', index=False)
    print(f"  Saved: leta_binned_trajectories.csv")

    # Step 3: Environment profiles
    print("[Step 3] Computing environment profiles...")
    profiles = compute_environment_profiles(df)
    profiles.to_csv(outdir / 'leta_environment_profiles.csv', index=False)
    print(f"  Saved: leta_environment_profiles.csv")

    # Step 4: Stability metrics
    print("[Step 4] Computing stability metrics...")
    stability = compute_stability_metrics(binned)
    stability.to_csv(outdir / 'leta_stability_metrics.csv', index=False)
    print(f"  Saved: leta_stability_metrics.csv")

    # Step 5: Cross-environment comparisons
    print("[Step 5] Computing cross-environment comparisons...")
    comparisons = compute_cross_environment_tests(df)
    comparisons.to_csv(outdir / 'leta_cross_environment_comparison.csv', index=False)
    print(f"  Saved: leta_cross_environment_comparison.csv")

    # Step 6: Generate figures
    print("\n[Step 6] Generating figures...")
    plot_radar(profiles, outdir / 'Figure_LETA_radar.png')
    plot_trajectories(binned, outdir / 'Figure_LETA_trajectories.png')
    plot_heatmap(stability, outdir / 'Figure_LETA_heatmap.png')

    # Step 7: Manuscript tables
    print("\n[Step 7] Generating manuscript tables...")
    generate_manuscript_table(stability, comparisons, outdir / 'leta_manuscript_table.txt')

    # Final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\nKey outputs:")
    print(f"  Data:    leta_binned_trajectories.csv")
    print(f"  Data:    leta_environment_profiles.csv")
    print(f"  Data:    leta_stability_metrics.csv")
    print(f"  Data:    leta_cross_environment_comparison.csv")
    print(f"  Figure:  Figure_LETA_radar.png")
    print(f"  Figure:  Figure_LETA_trajectories.png")
    print(f"  Figure:  Figure_LETA_heatmap.png")
    print(f"  Table:   leta_manuscript_table.txt")

    # Print key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    # Most differentiated emotions across environments
    sig_comparisons = comparisons[comparisons['p_value'] < 0.05]
    if len(sig_comparisons) > 0:
        print(f"\nSignificant cross-environment differences: {len(sig_comparisons)}")
        largest = sig_comparisons.loc[sig_comparisons['cohens_d'].abs().idxmax()]
        print(f"  Largest effect: {largest['emotion']} between "
              f"{largest['env_1']} and {largest['env_2']} "
              f"(d = {largest['cohens_d']:.3f}, p = {largest['p_value']:.4f})")
    else:
        print("\nNo significant cross-environment differences at p < .05")

    # Stability summary
    high_cv = stability[stability['cv'].notna() & (stability['cv'] > 0.5)]
    print(f"\nHigh-volatility indicators (CV > 0.5): {len(high_cv)}")
    stable = stability[stability['cv'].notna() & (stability['cv'] <= 0.3)]
    print(f"Stable indicators (CV <= 0.3): {len(stable)}")


if __name__ == '__main__':
    main()