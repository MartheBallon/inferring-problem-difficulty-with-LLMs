from typing import List, Dict

import random
import pandas as pd
import networkx as nx

from ortools.sat.python import cp_model


def create_all_pairs_from_df(
        df: pd.DataFrame,
        column_mapping: Dict[str, str],
        seed: int = 42,
    ) -> pd.DataFrame:
    """
    column_mapping example: 
    {
        "id": "question_number",
        "difficulty": "question_diff",
        "problem": "full_question"
    }
    """
    random.seed(seed)

    n = len(df)
    k = len(df) - 1

    graph = nx.random_regular_graph(k, n, seed=42)
    edges = list(graph.edges())

    random.shuffle(edges)

    pairs = []
    for i, j in edges:
        pair = {}
        for key, col_name in column_mapping.items():
            pair[f"{key}_1"] = df.at[i, col_name]
            pair[f"{key}_2"] = df.at[j, col_name]
        pairs.append(pair)

    return pd.DataFrame(pairs)


def incremental_samples(
    df_pairs: pd.DataFrame,
    id_columns: List[str],
    k_values: List[int],
) -> Dict[int, pd.DataFrame]:
    """
    Exact incremental b‑matching with nested edge sets.

    Parameters
    ----------
    df_pairs   : DataFrame with two identifier columns `id_columns`
    id_columns : [col_A, col_B]  (order is irrelevant)
    k_values   : list[int].  Must be positive; will be solved in ascending order.

    Returns
    -------
    dict[k] -> DataFrame  (same columns as input plus
                           'full_id'  =  f'{A}_{B}',
                           'first_round' = k when the edge first appeared)
    """
    id_col1, id_col2 = id_columns
    # All unique endpoints
    vertices = pd.unique(pd.concat([df_pairs[id_col1], df_pairs[id_col2]]))
    # Incident‑edge index lists for degree constraints
    incident = {v: [] for v in vertices}
    for idx, v1, v2 in df_pairs[[id_col1, id_col2]].itertuples(index=True, name=None):
        incident[v1].append(idx)
        incident[v2].append(idx)

    edge_indices = df_pairs.index.tolist()
    # Book‑keeping over successive k
    selected_prev: set[int] = set()
    first_round: dict[int, int] = {}
    results: dict[int, pd.DataFrame] = {}

    for k in sorted(k_values):
        # ---------- build model ----------
        m = cp_model.CpModel()
        x = {idx: m.NewBoolVar(f"x_{idx}") for idx in edge_indices}

        # Lock edges chosen in earlier rounds  (nesting constraint)
        for idx in selected_prev:
            m.Add(x[idx] == 1)

        # Degree ≤ k on every vertex
        for v in vertices:
            m.Add(sum(x[idx] for idx in incident[v]) <= k)

        # Maximise number of edges
        m.Maximize(sum(x.values()))

        # ---------- solve ----------
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = 123456
        solver.parameters.num_search_workers = 1

        status = solver.Solve(m)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"CP‑SAT failed with status {solver.StatusName(status)}")

        # ---------- extract solution ----------
        selected_now = [idx for idx in edge_indices if solver.Value(x[idx])]
        for idx in selected_now:
            if idx not in first_round:
                first_round[idx] = k
        selected_prev = set(selected_now)

        df_sel = df_pairs.loc[selected_now].copy()
        df_sel.insert(0, "first_round", [first_round[idx] for idx in selected_now])
        df_sel.insert(
            0, "full_id", df_sel.apply(lambda r: f"{r[id_col1]}_{r[id_col2]}", axis=1)
        )
        df_sel.reset_index(drop=True, inplace=True)
        results[k] = df_sel

    return results
