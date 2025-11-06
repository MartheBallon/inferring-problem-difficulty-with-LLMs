import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
from src.prompts import DIFFICULTY_PROMPT
# from src.utils import write_jsonl

df = pd.read_parquet("data/jee/jee_pairs.parquet")

messages = []
for i, row in df.iterrows():
    messages.append(
        {
            "custom_id": f"{row['id_1']}_{row['id_2']}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "o3-2025-04-16",
                "reasoning": {
                    "effort": "medium",
                },
                "instructions": DIFFICULTY_PROMPT,
                "input": f"Problem a: {row['problem_1']}\nProblem b: {row['problem_2']}",
            },
        }
    )

# write_jsonl("data/jee/jee_pairs_o3.jsonl", messages)

messages = []
for i, row in df.iterrows():
    req = {
        "systemInstruction": {
            "parts": [{"text": DIFFICULTY_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": f"Problem a: {row['problem_1']}\nProblem b: {row['problem_2']}"
                }]
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 8192,
            }
        }
    }
    line = {"key": f"{row['id_1']}_{row['id_2']}", "request": req}
    messages.append(line)

# write_jsonl("data/jee/jee_pairs_gemini.jsonl", messages)

# upload Gemini batch
# from google import genai
# from google.genai import types

# client = genai.Client()

# uploaded = client.files.upload(
#     file="data/jee/jee_pairs_gemini.jsonl",
#     config=types.UploadFileConfig(mime_type="jsonl"),
# )
# job = client.batches.create(
#     model="gemini-2.5-pro",
#     src=uploaded.name,
# )
