import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr

df = pd.read_parquet("results/omni/omni_with_bt.parquet")

iterations = 1000
Rseed = 42
results = []

#o3 correlations
for i in range(iterations):
    subset_df = df.sample(n=800, random_state=Rseed+i)
    # Compute correlations
    pearson_corr = pearsonr(subset_df["difficulty"], subset_df["o3_bt"])[0]
    kendall_corr = kendalltau(subset_df["difficulty"], subset_df["o3_bt"])[0]
    spearman_corr = spearmanr(subset_df["difficulty"], subset_df["o3_bt"])[0]
    # Store results
    results.append({
        "iteration": i,
        "pearson": pearson_corr,
        "kendall": kendall_corr,
        "spearman": spearman_corr
    })

results_df = pd.DataFrame(results)
results_df.to_parquet("results/omni/omni_subsample_o3_correlations.parquet", index=False)


#gemini correlations
results = []
for i in range(iterations):
    subset_df = df.sample(n=800, random_state=Rseed+i)
    # Compute correlations
    pearson_corr = pearsonr(subset_df["difficulty"], subset_df["gemini_bt"])[0]
    kendall_corr = kendalltau(subset_df["difficulty"], subset_df["gemini_bt"])[0]
    spearman_corr = spearmanr(subset_df["difficulty"], subset_df["gemini_bt"])[0]
    # Store results
    results.append({
        "iteration": i,
        "pearson": pearson_corr,
        "kendall": kendall_corr,
        "spearman": spearman_corr
    })

results_df = pd.DataFrame(results)
results_df.to_parquet("results/omni/omni_subsample_gemini_correlations.parquet", index=False)
