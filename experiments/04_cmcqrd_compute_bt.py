import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.bt import compute_bt_ratings

df = pd.read_parquet("data/cmcqrd/cmcqrd.parquet")
pairs_results = pd.read_parquet("results/cmcqrd/cmcqrd_pairs_results.parquet")

pairs_o3 = pairs_results[["id_1", "id_2", "o3"]]
pairs_o3.columns = ["id_1", "id_2", "a_wins"]
o3_strengths = compute_bt_ratings(pairs_o3)

pairs_gemini = pairs_results[["id_1", "id_2", "gemini"]]
pairs_gemini.columns = ["id_1", "id_2", "a_wins"]
gemini_strengths = compute_bt_ratings(pairs_gemini)

df.set_index("question_number", inplace=True)

df = df.join(
    pd.DataFrame(
        o3_strengths.to_list(),
        columns=["o3_bt"],
        index=o3_strengths.index,
    ),
    how='left',
)
df = df.join(
    pd.DataFrame(
        gemini_strengths.to_list(),
        columns=["gemini_bt"],
        index=gemini_strengths.index,
    ),
    how='left',
)

df.reset_index(drop=False, inplace=True)

# df.to_parquet("results/cmcqrd/cmcqrd_with_bt.parquet", index=False)
