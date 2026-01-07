import os
import asyncio
import json
import random
from env import Actor2 as act
from typing import List, Dict, Any

def preprocess(result):
    data = result['extra']
    data['reward'] = result['score']
    return data

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    """
    Write a list of dictionaries to a JSONL file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


actor = act()

cnt_per_task = 1000

for task_type in range(5):
    succ, total = 0, cnt_per_task
    for i in range(cnt_per_task):
        task_id = random.randint(0, 20000000)
        task_id = task_id * 5 + task_type

        result = asyncio.run(actor.evaluate(task_id=task_id))

        print(f"{result['task_name'].split(":")[1]}:{task_id}:{result['score']}")

        succ += result['score']

        if result['score'] == 1.0:
            write_jsonl(f"dataset/goofspiel_{8+task_type*2}.jsonl", [preprocess(result)])

    print(f"{succ}/{total} = {succ/total*100}%")
