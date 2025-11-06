import json
from typing import Dict, List
import numpy as np
import pandas as pd


def write_jsonl(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def load_jsonl(filepath) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    

def assign_random_score(df_entry):
    label = np.random.randint(1, 11)
    return int(label)

def flip_score(df_entry):
    if df_entry == 1:
        return 0
    elif df_entry == 0:
        return 1
    
# parse answer omni-judge
def parse_report(report):
    parts = report.split("## ")
    data = {}
    
    for part in parts[1:]:  
        lines = part.strip().split("\n")
        title = lines[0].strip() 
        content = "\n".join(lines[1:]).strip()  
        
        if title == "Justification":
            data[title] = content
        elif title == "Student Final Answer":
            data[title] = content
        else:
            data[title] = lines[1].strip() if len(lines) > 1 else ''
    
    return data

def get_dataframe_reasoning_models(file):
    records = []
    count = 0
    with open("Omni-Math-2-Algebra.jsonl", "r") as ref:
        lines = ref.readlines()
        with open(file, 'r') as file:
            for idx, line in enumerate(file):
                count += 1
                json_obj = json.loads(line)
                ref_object = json.loads(lines[idx])
                info = parse_report(json_obj['omni-judge'])
                if info == {}:
                    continue
                try:
                    correctness = info['Equivalence Judgement']
                    if correctness == 'TRUE':
                        records.append({'id': ref_object['id'], 'difficulty': json_obj['difficulty'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'o3_final_answer': info['Student Final Answer'], 'o3_score': 1})
                    else:
                        records.append({'id': ref_object['id'], 'difficulty': json_obj['difficulty'], 'problem': json_obj['problem'], 'answer': json_obj['answer'], 'o3_final_answer': info['Student Final Answer'], 'o3_score': 0})
                except:
                    continue
        
    Data_df = pd.DataFrame(records)
    return Data_df

