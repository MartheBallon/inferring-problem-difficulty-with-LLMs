import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.pairs import create_all_pairs_from_df


df = pd.read_parquet("data/jee/jee.parquet")

df_pairs = create_all_pairs_from_df(
    df,
    column_mapping={
        "id": "Question Number",
        "difficulty": "Difficulty",
        "problem": "Question"
    }
)

# df_pairs.to_parquet(
#     "data/jee/jee_pairs.parquet",
#     index=False,
# )
