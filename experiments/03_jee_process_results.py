import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd


PATTERN_o3 = r"'text':\s*'@<([^>]+)>@'"
PATTERN_Gemini = r"@<([ab])>@"

pairs = pd.read_parquet("data/jee/jee_pairs.parquet")
pairs.insert(0, "full_id", pairs.id_1.astype(str) + "_" + pairs.id_2.astype(str))

pairs.set_index(["full_id"], inplace=True)

pair_results_o3 = pd.read_json(
    "results/jee/jee_pairs_o3_results.jsonl",
    lines=True,
    dtype={"response": str, 'custom_id': str},
)
pair_results_gemini = pd.read_json(
    "results/jee/jee_pairs_gemini_results.jsonl",
    lines=True,
    dtype={"response": str, 'key': str},
)

pairs = pairs.join(
    pd.DataFrame(
        pair_results_o3['response'].str.extract(PATTERN_o3, expand=False).to_list(),
        index=pair_results_o3["custom_id"].to_list(),
        columns=["o3"],
    )
)
pairs = pairs.join(
    pd.DataFrame(
        pair_results_gemini['response'].str.extract(PATTERN_Gemini, expand=False).to_list(),
        index=pair_results_gemini["key"].to_list(),
        columns=["gemini"],
    )
)

pairs.replace({"o3": {"a": 1, "b": 0}, "gemini": {"a": 1, "b": 0}}, inplace=True)

pairs.reset_index(inplace=True)

# pairs.to_parquet(
#     "results/jee/jee_pairs_results.parquet",
#     index=False,
# )
