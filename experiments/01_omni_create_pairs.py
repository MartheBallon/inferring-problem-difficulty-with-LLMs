import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.pairs import create_all_pairs_from_df, incremental_samples


df = pd.read_parquet("data/omni/omni.parquet")

df_pairs = create_all_pairs_from_df(
    df,
    column_mapping={
        "id": "id",
        "difficulty": "difficulty",
        "problem": "problem"
    },
)

matches = incremental_samples(
    df_pairs,
    id_columns=["id_1","id_2"],
    k_values=[6,12,24,30,36,42,48,50,54,60,66,72,78,84,90,96,100,102,108,114,120,126,132,138,144,150,174,192,200],
)

# matches[200].to_parquet(
#     "data/omni/omni_pairs.parquet",
#     index=False,
# )
