#!/usr/bin/env python3
"""
clean_bccc_vuls_dataset.py

Analyze two CSV files (expected to be BCCC-VulSCs-2023 CSVs or similar feature matrices)
and produce statistics and charts for:
 - missing values (per-column and per-row)
 - null / NaN counts
 - total samples (per-file and combined)
 - feature types (dtypes and counts)
 - duplicates, constant columns, near-zero variance candidates
 - memory usage and sparsity
 - numeric distributions, correlations, and top correlated pairs
 - categorical cardinalities and top categories
 - basic class/label balance detection (if label-like columns exist)

Usage:
    python clean_bccc_vuls_dataset.py file1.csv file2.csv [--outdir analysis_output] [--show]

Examples:
    python clean_bccc_vuls_dataset.py train.csv test.csv
    python clean_bccc_vuls_dataset.py part1.csv part2.csv --outdir reports --show

Dependencies:
    pandas, numpy, matplotlib, seaborn, scipy (optional)
    Install with: pip install pandas numpy matplotlib seaborn scipy

Outputs:
    - Console summary
    - PNG charts saved in the output directory
    - summary JSON saved as summary_<timestamp>.json in the output directory

Author: GitHub Copilot (adapted)
"""

import argparse
import os
import sys
import json
import math
from datetime import datetime
from collections import Counter

# Try imports and provide helpful error if missing
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
except Exception as e:
    print("Error importing modules: {}".format(e), file=sys.stderr)
    print("Make sure you have pandas, numpy, matplotlib, seaborn and scipy installed.", file=sys.stderr)
    print("Install with: pip install pandas numpy matplotlib seaborn scipy", file=sys.stderr)
    sys.exit(1)

sns.set(style="whitegrid")

# Helpers
def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def safe_filename(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def detect_label_columns(df):
    # Heuristics to detect potential label columns
    candidates = []
    lower = [c.lower() for c in df.columns]
    for name in df.columns:
        n = name.lower()
        if n in ("label", "target", "class", "vulnerable", "y", "is_vulnerable"):
            candidates.append(name)
        elif "vul" in n and ("label" in n or "flag" in n or "is" in n):
            candidates.append(name)
    # If nothing obvious, check columns with 2 unique values or small unique set
    if not candidates:
        for name in df.columns:
            nunique = df[name].nunique(dropna=True)
            if nunique == 2:
                candidates.append(name)
    return candidates

def summarize_df_basic(df):
    summary = {}
    summary['num_rows'] = int(df.shape[0])
    summary['num_columns'] = int(df.shape[1])
    summary['memory_usage_bytes'] = int(df.memory_usage(deep=True).sum())
    summary['memory_usage_human'] = f"{summary['memory_usage_bytes'] / (1024**2):.2f} MB"
    summary['num_duplicates'] = int(df.duplicated().sum())
    # constant columns
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    summary['constant_columns'] = const_cols
    # missing stats
    missing_per_col = df.isna().sum()
    summary['columns_missing_count'] = int((missing_per_col > 0).sum())
    summary['rows_with_any_missing'] = int(df.isna().any(axis=1).sum())
    summary['percent_rows_with_missing'] = float(summary['rows_with_any_missing']) / max(1, summary['num_rows'])
    # dtype counts
    dtypes = df.dtypes.astype(str).value_counts().to_dict()
    summary['dtypes'] = dtypes
    return summary

def plot_missing_by_column(df, outpath, title_suffix=""):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values by column to plot.")
        return None
    plt.figure(figsize=(max(6, min(20, 0.2 * len(missing))), 6))
    sns.barplot(x=missing.values, y=missing.index, palette="viridis")
    plt.xlabel("Missing values (count)")
    plt.ylabel("Column")
    plt.title(f"Missing values per column {title_suffix}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def plot_missing_matrix(df, outpath, title_suffix=""):
    # Visualize missingness heatmap (rows x columns). For very large df, sample rows.
    nrows = df.shape[0]
    max_rows_for_plot = 1000
    if nrows > max_rows_for_plot:
        sample_df = df.sample(n=max_rows_for_plot, random_state=0)
    else:
        sample_df = df
    mat = sample_df.isna().T.astype(int)  # columns x rows
    plt.figure(figsize=(min(12, 0.02 * mat.shape[1] + 6), max(3, 0.02 * mat.shape[0] + 3)))
    sns.heatmap(mat, cmap="Greys", cbar=False)
    plt.xlabel("Sampled row index")
    plt.ylabel("Column")
    plt.title(f"Missingness matrix (sampled up to {max_rows_for_plot} rows) {title_suffix}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def plot_missing_row_counts(df, outpath, title_suffix=""):
    missing_per_row = df.isna().sum(axis=1)
    plt.figure(figsize=(8, 4))
    sns.histplot(missing_per_row, bins=50, kde=False)
    plt.xlabel("Missing values in row")
    plt.ylabel("Count of rows")
    plt.title(f"Distribution of missing values per row {title_suffix}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def plot_dtype_counts(df, outpath, title_suffix=""):
    dtypes = df.dtypes.astype(str).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=dtypes.index, y=dtypes.values, palette="pastel")
    plt.ylabel("Number of columns")
    plt.xlabel("Dtype")
    plt.title(f"Feature types {title_suffix}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def plot_numeric_distributions(df, outdir, prefix, max_cols=12):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        return []
    # pick most variable numeric columns
    variances = numeric.var(numeric_only=True).sort_values(ascending=False)
    selected = variances.index[:max_cols].tolist()
    outfiles = []
    for col in selected:
        series = numeric[col].dropna()
        if series.shape[0] == 0:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(series, bins=50, kde=True, color='C0')
        plt.xlabel(col)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        fname = os.path.join(outdir, f"{safe_filename(prefix)}_dist_{safe_filename(col)}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        outfiles.append(fname)
    return outfiles

def compute_top_correlations(df, top_k=20):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    # Use Spearman to be robust to non-normal distributions
    try:
        corr = numeric.corr(method='spearman')
    except Exception:
        corr = numeric.corr(method='pearson')
    corr_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            corr_pairs.append((cols[i], cols[j], float(val)))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return corr_pairs[:top_k], corr

def plot_correlation_heatmap(corr_df, outpath, title_suffix=""):
    if corr_df is None or corr_df.size == 0:
        return None
    # If too many features, clip to top N by absolute correlation variance
    n = corr_df.shape[0]
    max_n = 40
    if n > max_n:
        # select top columns by variance of correlations
        variances = corr_df.var().sort_values(ascending=False)
        selected = variances.index[:max_n]
        corr_plot = corr_df.loc[selected, selected]
    else:
        corr_plot = corr_df
    plt.figure(figsize=(min(16, 0.4 * corr_plot.shape[0] + 6), min(16, 0.4 * corr_plot.shape[1] + 6)))
    sns.heatmap(corr_plot, cmap='vlag', center=0, square=True, linewidths=0.1)
    plt.title(f"Correlation heatmap {title_suffix}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def analyze_categorical(df, outdir, prefix, max_unique_for_plot=20):
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    outfiles = []
    cat_summary = {}
    for col in cat_cols:
        nunique = df[col].nunique(dropna=True)
        cat_summary[col] = {'nunique': int(nunique)}
        if nunique <= max_unique_for_plot:
            counts = df[col].value_counts(dropna=False)
            plt.figure(figsize=(6, max(3, 0.3 * len(counts))))
            sns.barplot(y=counts.index.astype(str), x=counts.values, palette="Set3")
            plt.xlabel("Count")
            plt.ylabel(col)
            plt.title(f"Value counts for {col}")
            plt.tight_layout()
            fname = os.path.join(outdir, f"{safe_filename(prefix)}_cat_counts_{safe_filename(col)}.png")
            plt.savefig(fname, dpi=150)
            plt.close()
            outfiles.append(fname)
            # record top categories
            top = counts.head(10).to_dict()
            cat_summary[col]['top_values'] = {str(k): int(v) for k, v in top.items()}
    return cat_summary, outfiles

def find_near_zero_variance(df, freq_cut=95/5, unique_cut=10):
    # Basic heuristic: columns with very low variance or very imbalanced top categories
    nzv = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            nzv.append(col)
            continue
        nunique = s.nunique()
        if nunique <= 1:
            nzv.append(col)
            continue
        # for numeric: coefficient of variation
        if pd.api.types.is_numeric_dtype(s):
            if s.std() == 0:
                nzv.append(col)
                continue
            if abs(s.mean()) > 0:
                cv = s.std() / (abs(s.mean()) + 1e-12)
                if cv < 1e-6:
                    nzv.append(col)
                    continue
        # for categorical: check frequency ratio of most common to second most common
        top_counts = s.value_counts().values
        if len(top_counts) >= 2:
            ratio = top_counts[0] / (top_counts[1] + 1e-12)
            if ratio > freq_cut:
                nzv.append(col)
        elif len(top_counts) == 1:
            nzv.append(col)
    return nzv

def analyze_file(path, outdir, label="file"):
    df = pd.read_csv(path)
    name = os.path.splitext(os.path.basename(path))[0]
    prefix = f"{label}_{name}"
    info = {}
    info['path'] = path
    info['name'] = name
    info['shape'] = [int(df.shape[0]), int(df.shape[1])]
    info['summary'] = summarize_df_basic(df)

    # Missing plots
    mkdir_p(outdir)
    missing_col_png = os.path.join(outdir, f"{safe_filename(prefix)}_missing_by_column.png")
    m1 = plot_missing_by_column(df, missing_col_png, title_suffix=f"({name})")
    if m1:
        info.setdefault('plots', []).append(m1)

    missing_matrix_png = os.path.join(outdir, f"{safe_filename(prefix)}_missing_matrix.png")
    m2 = plot_missing_matrix(df, missing_matrix_png, title_suffix=f"({name})")
    if m2:
        info.setdefault('plots', []).append(m2)

    missing_row_png = os.path.join(outdir, f"{safe_filename(prefix)}_missing_by_row.png")
    m3 = plot_missing_row_counts(df, missing_row_png, title_suffix=f"({name})")
    if m3:
        info.setdefault('plots', []).append(m3)

    # dtype plot
    dtype_png = os.path.join(outdir, f"{safe_filename(prefix)}_dtypes.png")
    m4 = plot_dtype_counts(df, dtype_png, title_suffix=f"({name})")
    if m4:
        info.setdefault('plots', []).append(m4)

    # numeric distributions
    dist_files = plot_numeric_distributions(df, outdir, prefix, max_cols=12)
    info.setdefault('plots', []).extend(dist_files)

    # correlations
    top_corr_pairs, corr_df = compute_top_correlations(df, top_k=30)
    info['top_correlations'] = [{'col1': a, 'col2': b, 'corr': c} for (a,b,c) in top_corr_pairs]
    corr_png = os.path.join(outdir, f"{safe_filename(prefix)}_correlation_heatmap.png")
    m5 = plot_correlation_heatmap(corr_df, corr_png, title_suffix=f"({name})")
    if m5:
        info.setdefault('plots', []).append(m5)

    # categorical summary
    cat_summary, cat_plots = analyze_categorical(df, outdir, prefix, max_unique_for_plot=30)
    info['categorical_summary'] = cat_summary
    info.setdefault('plots', []).extend(cat_plots)

    # near-zero variance
    nzv = find_near_zero_variance(df)
    info['near_zero_variance_candidates'] = nzv

    # potential label columns
    labels = detect_label_columns(df)
    info['label_candidates'] = labels
    if labels:
        # show class balance for each detected label
        info['label_balance'] = {}
        for label_col in labels:
            counts = df[label_col].value_counts(dropna=False).to_dict()
            info['label_balance'][label_col] = {str(k): int(v) for k, v in counts.items()}

    # top missing columns
    info['top_missing_columns'] = df.isna().sum().sort_values(ascending=False).head(20).to_dict()

    return df, info

def compare_two_frames(df1, df2):
    # Compare shapes, overlapping columns, columns present only in one, basic stats differences for shared numeric columns
    comp = {}
    comp['shape_left'] = [int(df1.shape[0]), int(df1.shape[1])]
    comp['shape_right'] = [int(df2.shape[0]), int(df2.shape[1])]
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    comp['common_columns'] = sorted(list(cols1 & cols2))
    comp['left_only_columns'] = sorted(list(cols1 - cols2))
    comp['right_only_columns'] = sorted(list(cols2 - cols1))
    # Compare missingness for common columns
    common = comp['common_columns']
    miss1 = df1[common].isna().sum()
    miss2 = df2[common].isna().sum()
    miss_comp = []
    for c in common:
        miss_comp.append({'column': c, 'missing_left': int(miss1[c]), 'missing_right': int(miss2[c])})
    comp['missing_comparison'] = sorted(miss_comp, key=lambda x: (abs(x['missing_left'] - x['missing_right'])), reverse=True)[:50]
    # numeric summary differences
    numeric_common = [c for c in common if pd.api.types.is_numeric_dtype(df1[c]) or pd.api.types.is_numeric_dtype(df2[c])]
    num_comp = []
    for c in numeric_common:
        s1 = df1[c].dropna()
        s2 = df2[c].dropna()
        if s1.empty or s2.empty:
            continue
        stats1 = {'mean': float(s1.mean()), 'std': float(s1.std()), 'median': float(s1.median())}
        stats2 = {'mean': float(s2.mean()), 'std': float(s2.std()), 'median': float(s2.median())}
        num_comp.append({'column': c, 'left': stats1, 'right': stats2})
    comp['numeric_stats_comparison'] = num_comp[:200]
    return comp

def save_summary_json(summary, outdir):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = os.path.join(outdir, f"summary_{timestamp}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return fname

def main():
    parser = argparse.ArgumentParser(description="Analyze two CSV files and produce missing value, dtype, and other diagnostics + charts.")
    parser.add_argument("file1", help="Path to first CSV file")
    parser.add_argument("file2", help="Path to second CSV file")
    parser.add_argument("--outdir", default="analysis_output", help="Directory to save charts and summary (default: analysis_output)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (useful in notebooks/local).")
    parser.add_argument("--sample", type=int, default=0, help="If >0, sample this many rows from each file before analysis (useful for very large files).")
    args = parser.parse_args()

    mkdir_p(args.outdir)

    print(f"Loading file1: {args.file1}")
    try:
        df1 = pd.read_csv(args.file1)
    except Exception as e:
        print(f"Failed to read {args.file1}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded file1 shape: {df1.shape}")

    print(f"Loading file2: {args.file2}")
    try:
        df2 = pd.read_csv(args.file2)
    except Exception as e:
        print(f"Failed to read {args.file2}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded file2 shape: {df2.shape}")

    if args.sample and args.sample > 0:
        print(f"Sampling up to {args.sample} rows from each file for lightweight analysis.")
        df1 = df1.sample(n=min(args.sample, df1.shape[0]), random_state=0)
        df2 = df2.sample(n=min(args.sample, df2.shape[0]), random_state=0)

    print("Analyzing first file...")
    df1_full, info1 = analyze_file(args.file1, args.outdir, label="left")
    print("Analyzing second file...")
    df2_full, info2 = analyze_file(args.file2, args.outdir, label="right")

    print("Comparing the two files...")
    comp = compare_two_frames(df1_full, df2_full)

    # Aggregate summary
    summary = {
        'analyzed_at_utc': datetime.utcnow().isoformat() + "Z",
        'file_left': info1,
        'file_right': info2,
        'comparison': comp
    }

    summary_path = save_summary_json(summary, args.outdir)
    print(f"Saved JSON summary to: {summary_path}")

    # Print a readable console summary
    def print_basic_info(label, info):
        print(f"\n--- {label} ({info.get('name')}) ---")
        print(f"Path: {info.get('path')}")
        print(f"Shape: {info.get('shape')}")
        print(f"Memory usage: {info.get('summary', {}).get('memory_usage_human')}")
        print(f"Columns with missing values: {info.get('summary', {}).get('columns_missing_count')}")
        print(f"Rows with any missing values: {info.get('summary', {}).get('rows_with_any_missing')} ({info.get('summary', {}).get('percent_rows_with_missing')*100:.2f}%)")
        print(f"Duplicate rows: {info.get('summary', {}).get('num_duplicates')}")
        print(f"Constant columns: {len(info.get('summary', {}).get('constant_columns', []))}")
        if info.get('label_candidates'):
            print(f"Potential label columns detected: {info.get('label_candidates')}")
            for lc in info.get('label_candidates', []):
                print(f"  Balance for {lc}: {info.get('label_balance', {}).get(lc)}")

    print_basic_info("File 1", info1)
    print_basic_info("File 2", info2)

    print("\nTop differences in missingness between the two files (up to 50 shown):")
    for d in comp.get('missing_comparison', [])[:50]:
        print(f"  {d['column']}: missing_left={d['missing_left']}, missing_right={d['missing_right']}")

    print("\nTop correlated numeric column pairs (file1):")
    for item in info1.get('top_correlations', [])[:10]:
        print(f"  {item['col1']} <> {item['col2']}, corr={item['corr']:.3f}")

    print("\nTop correlated numeric column pairs (file2):")
    for item in info2.get('top_correlations', [])[:10]:
        print(f"  {item['col1']} <> {item['col2']}, corr={item['corr']:.3f}")

    print(f"\nAll charts and outputs saved to: {os.path.abspath(args.outdir)}")
    if args.show:
        print("Showing generated PNG files is not implemented in this script; set --show to display inline when run in an environment that supports it.")
    print("Done.")

if __name__ == "__main__":
    main()