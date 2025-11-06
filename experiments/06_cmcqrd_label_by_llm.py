import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.prompts import LABEL_PROMPT
from src.utils import write_jsonl

df = pd.read_parquet("data/cmcqrd/cmcqrd.parquet")

messages = []
for i, row in df.iterrows():
    messages.append(
        {
            "custom_id": f"{row['question_number']}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "o3-2025-04-16",
                "reasoning": {
                    "effort": "medium",
                },
                "instructions": LABEL_PROMPT,
                "input": f"{row['full_question']}",
            },
        }
    )

write_jsonl("data/cmcqrd/cmcqrd_label_03.jsonl", messages)

messages = []
for i, row in df.iterrows():
    req = {
        "systemInstruction": {
            "parts": [{"text": LABEL_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": f"{row['full_question']}"
                }]
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingBudget": 8192,
            }
        }
    }
    line = {"key": f"{row['question_number']}", "request": req}
    messages.append(line)

write_jsonl("data/cmcqrd/cmcqrd_label_gemini.jsonl", messages)