import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.prompts import SOLVE_PROMPT_OMNI
from src.utils import write_jsonl

df = pd.read_parquet("data/omni/omni.parquet")

messages = []
for i, row in df.iterrows():
    req = {
        "systemInstruction": {
            "parts": [{"text": SOLVE_PROMPT_OMNI}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": f"{row['problem']}"
                }]
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 8192,
            }
        }
    }
    line = {"key": f"{row['id']}", "request": req}
    messages.append(line)

write_jsonl("data/omni/omni_benchmark.jsonl", messages)

# from google import genai
# from google.genai import types

# client = genai.Client()

# uploaded = client.files.upload(
#     file="data/omni/omni_benchmark.jsonl",
#     config=types.UploadFileConfig(mime_type="jsonl"),
# )
# job = client.batches.create(
#     model="gemini-2.5-pro",
#     src=uploaded.name,
# )
