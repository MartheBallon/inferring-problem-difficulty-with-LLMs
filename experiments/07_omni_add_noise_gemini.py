import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr

from src.utils import assign_random_score, flip_score
from src.bt import compute_bt_ratings

alpha = 0.1  # Noise percentage
iteration = 1000
Rseed = 42

# Add noise to BT score == switch outcome of % pair matches 
df_omni = pd.read_parquet("results/omni/omni_pairs_results.parquet")
num_flips = int(alpha * len(df_omni))

df_original = pd.read_parquet("results/omni/omni_with_bt.parquet")

pearson_corr = []
kendall_corr = []
spearman_corr = []
for it in range(iteration):
    randomgen = np.random.default_rng(seed=Rseed + it)
    flip_indices = randomgen.choice(df_omni.index, size=num_flips, replace=False)
    df_new = df_omni.copy()
    df_new.loc[flip_indices, 'gemini'] = df_new.loc[flip_indices, 'gemini'].apply(flip_score)
    df_new['gemini'] = df_new['gemini'].astype(float)

    pairs_gemini = df_new[["id_1", "id_2", "gemini"]]
    pairs_gemini.columns = ["id_1", "id_2", "a_wins"]
    gemini_strengths = compute_bt_ratings(pairs_gemini).to_list()

    gemini_bt_original = df_original['gemini_bt'].astype(float).to_numpy()
    gemini_bt_noisy = np.array(gemini_strengths)

    corr_pearson = pearsonr(gemini_bt_original, gemini_bt_noisy)[0]
    corr_kendall = kendalltau(gemini_bt_original, gemini_bt_noisy)[0]
    corr_spearman = spearmanr(gemini_bt_original, gemini_bt_noisy)[0]

    pearson_corr.append(corr_pearson)
    kendall_corr.append(corr_kendall)
    spearman_corr.append(corr_spearman)

records = [{'it': it, 'pearson': pearson, 'kendall': kendall, 'spearman': spearman, 'type': 'bt'} for it, pearson, kendall, spearman in zip(range(iteration), pearson_corr, kendall_corr, spearman_corr)]
df_results_bt = pd.DataFrame(records)

#df_omni.to_parquet(f"results/omni/omni_pairs_o3_noisy_alpha_{alpha}_it_{iteration}.parquet")

# Add noise to LLM performance
df_original = pd.read_parquet("results/omni/omnimath_algebra.parquet") 
df_omni = pd.read_parquet("results/omni/omnimath_algebra.parquet")
num_flips = int(alpha * len(df_omni))

pearson_corr = []
kendall_corr = []
spearman_corr = []
for it in range(iteration):
    randomgen = np.random.default_rng(seed=Rseed + it)
    flip_indices = randomgen.choice(df_omni.index, size=num_flips, replace=False)
    df_new = df_omni.copy()
    df_new.loc[flip_indices, 'gemini_score'] = df_new.loc[flip_indices, 'gemini_score'].apply(flip_score)

    performance_original = df_original['gemini_score'].astype(float).to_numpy()
    performance_noisy = df_new['gemini_score'].astype(float).to_numpy()

    corr_pearson = pearsonr(performance_original, performance_noisy)[0]
    corr_kendall = kendalltau(performance_original, performance_noisy)[0]
    corr_spearman = spearmanr(performance_original, performance_noisy)[0]

    pearson_corr.append(corr_pearson)
    kendall_corr.append(corr_kendall)
    spearman_corr.append(corr_spearman)

records = [{'it': it, 'pearson': pearson, 'kendall': kendall, 'spearman': spearman, 'type': 'performance'} for it, pearson, kendall, spearman in zip(range(iteration), pearson_corr, kendall_corr, spearman_corr)]
df_results_performance = pd.DataFrame(records)


# Add noise to LLM labelling
df_omni = pd.read_parquet("results/omni/omni_with_labels.parquet")
num_flips = int(alpha * len(df_omni))
df_original = pd.read_parquet("results/omni/omni_with_labels.parquet")

pearson_corr = []
kendall_corr = []
spearman_corr = []
for it in range(iteration):
    randomgen = np.random.default_rng(seed=Rseed + it)
    flip_indices = randomgen.choice(df_omni.index, size=num_flips, replace=False)
    df_new = df_omni.copy()
    df_new.loc[flip_indices, 'gemini_label'] = df_new.loc[flip_indices, 'gemini_label'].apply(assign_random_score)

    labels_original = df_original['gemini_label'].astype(float).to_numpy()
    labels_noisy = df_new['gemini_label'].astype(float).to_numpy()

    corr_pearson = pearsonr(labels_original, labels_noisy)[0]
    corr_kendall = kendalltau(labels_original, labels_noisy)[0]
    corr_spearman = spearmanr(labels_original, labels_noisy)[0]

    pearson_corr.append(corr_pearson)
    kendall_corr.append(corr_kendall)
    spearman_corr.append(corr_spearman)

records = [{'it': it, 'pearson': pearson, 'kendall': kendall, 'spearman': spearman, 'type': 'labels'} for it, pearson, kendall, spearman in zip(range(iteration), pearson_corr, kendall_corr, spearman_corr)]
df_results_labels = pd.DataFrame(records)


# Join DataFrames
df_results = pd.concat([df_results_bt, df_results_performance, df_results_labels], axis=0)
df_results.to_parquet(f"results/omni/omni_gemini_noise_alpha_{alpha}.parquet", index=False)