import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict

# Font settings for editable SVG text
matplotlib.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']
matplotlib.rcParams['svg.fonttype'] = 'none'

# Reversed region order
CUSTOM_REGION_ORDER = [
    "VISp", "VISpor", "VISpl", "VISli", "VISl", "VISal", "VISrl",
    "VISa", "VISam", "VISpm", "RSPagl", "RSPd", "RSPv"
]

HEMISPHERE_LABELS = {
    1: "Left",
    2: "Right",
    3: "Midline"
}

def make_plot(df, region_cols, output_prefix, title):
    means = df[region_cols].mean()
    sems = df[region_cols].sem()

    plt.figure(figsize=(14, 6))
    x = np.arange(len(region_cols))
    plt.bar(x, means[region_cols], yerr=sems[region_cols], capsize=4, color='skyblue', edgecolor='black')

    for i, col in enumerate(region_cols):
        y_vals = df[col].values
        x_vals = np.random.normal(loc=x[i], scale=0.05, size=len(y_vals))
        plt.scatter(x_vals, y_vals, color='black', alpha=0.6, s=10)

    plt.xticks(ticks=x, labels=region_cols, rotation=45, ha='right')
    plt.ylabel("Mean Value ± SEM")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_prefix + ".png", dpi=300)
    plt.savefig(output_prefix + ".svg")
    plt.close()
    print(f"✅ Saved: {output_prefix}.png/.svg")

def make_hemisphere_comparison_plot(df_all, region_cols, output_prefix, title):
    if 'hemisphere_id' not in df_all.columns:
        print(f"⚠️ No 'hemisphere_id' column found; skipping hemisphere comparison plot.")
        return

    grouped = df_all.groupby('hemisphere_id')
    hemisphere_dfs = {k: v for k, v in grouped if k in HEMISPHERE_LABELS}

    if not hemisphere_dfs:
        print(f"⚠️ No matching hemispheres (1, 2, 3) found in data.")
        return

    bar_width = 0.25
    x = np.arange(len(region_cols))

    plt.figure(figsize=(16, 6))

    for i, h_id in enumerate([1, 2, 3]):
        if h_id not in hemisphere_dfs:
            continue
        h_df = hemisphere_dfs[h_id]
        means = h_df[region_cols].mean()
        sems = h_df[region_cols].sem()
        offset = (i - 1) * bar_width
        plt.bar(x + offset, means[region_cols], bar_width, yerr=sems[region_cols],
                label=HEMISPHERE_LABELS[h_id], capsize=3)

        for j, col in enumerate(region_cols):
            if col in h_df.columns:
                y_vals = h_df[col].values
                x_vals = np.random.normal(loc=x[j] + offset, scale=0.05, size=len(y_vals))
                plt.scatter(x_vals, y_vals, alpha=0.6, s=10, label=None, color='black')

    plt.xticks(ticks=x, labels=region_cols, rotation=45, ha='right')
    plt.ylabel("Mean Value ± SEM")
    plt.title(title + " (Hemispheric Comparison)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_prefix + "_hemisphere_comparison.png", dpi=300)
    plt.savefig(output_prefix + "_hemisphere_comparison.svg")
    plt.close()
    print(f"✅ Saved: {output_prefix}_hemisphere_comparison.png/.svg")

def process_all_csvs(input_dir):
    grouped_files = defaultdict(list)

    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".csv") and not fname.startswith("._"):
            base_key = re.sub(r'_hemisphere_\d_', '_hemisphere_X_', fname)
            grouped_files[base_key].append(fname)

    for base_key, file_group in grouped_files.items():
        dfs = []
        region_cols_all = set()

        for fname in file_group:
            fpath = os.path.join(input_dir, fname)
            df = pd.read_csv(fpath)
            match = re.search(r'_hemisphere_(\d)_', fname)
            if match:
                df['hemisphere_id'] = int(match.group(1))
            else:
                print(f"⚠️ Could not find hemisphere ID in filename: {fname}")
                continue

            if "filename" in df.columns:
                df = df.drop(columns=["filename"])

            region_cols = [col for col in CUSTOM_REGION_ORDER if col in df.columns]
            region_cols_all.update(region_cols)
            dfs.append(df)

            # Still generate individual hemisphere-specific plot
            base_name = os.path.splitext(os.path.basename(fname))[0]
            title_base = base_name.replace("_", " ")
            output_prefix_full = os.path.join(input_dir, base_name)
            make_plot(df, region_cols, output_prefix_full, title_base)
            region_cols_no_visp = [r for r in region_cols if r != "VISp"]
            if region_cols_no_visp:
                make_plot(df, region_cols_no_visp, output_prefix_full + "_NO-VISp", title_base + " (No VISp)")

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            region_cols_all = [col for col in CUSTOM_REGION_ORDER if col in region_cols_all]

            # Generate combined hemisphere comparison plot for this data type
            common_prefix = base_key.replace("_hemisphere_X_", "_").replace(".csv", "")
            out_prefix = os.path.join(input_dir, common_prefix)
            make_hemisphere_comparison_plot(df_all, region_cols_all, out_prefix, common_prefix.replace("_", " "))

            region_cols_no_visp = [r for r in region_cols_all if r != "VISp"]
            if region_cols_no_visp:
                make_hemisphere_comparison_plot(df_all, region_cols_no_visp, out_prefix + "_NO-VISp", common_prefix.replace("_", " ") + " (No VISp)")

def main(input_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    process_all_csvs(input_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bar plots from CSVs")
    parser.add_argument("--input_dir", required=True, help="Directory containing CSV files")
    args = parser.parse_args()
    main(args.input_dir)
