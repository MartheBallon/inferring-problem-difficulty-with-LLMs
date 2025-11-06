import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.pairs import create_all_pairs_from_df, incremental_samples


df = pd.read_parquet("data/cmcqrd/cmcqrd.parquet")

df_pairs = create_all_pairs_from_df(
    df,
    column_mapping={
        "id": "question_number",
        "difficulty": "question_diff",
        "problem": "full_question"
    },
)

matches = incremental_samples(
    df_pairs,
    id_columns=["id_1","id_2"],
    k_values=[6,12,24,30,36,42,48,50,54,60,66],
)

# matches[66].to_parquet(
#     "data/cmcqrd/cmcqrd_pairs.parquet",
#     index=False,
# )
