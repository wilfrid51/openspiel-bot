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

task_type = [0, 0]

succ, total = 0, 1000

for i in range(total):
    id = random.randint(0, 1)
    task_id = 0
    while True:
        task_id = task_type[id] * 100000000 + random.randint(0, 100000000)
        if task_id % 5 == 0:
            break
    print(f"{'='*10} {task_id} {'='*10}")
    result = asyncio.run(actor.evaluate(task_id=task_id))

    print(f"{result['task_name'].split(":")[1]}:{task_id}:{result['score']}")
    succ += result['score']
    write_jsonl("goofspiel_8.jsonl", [preprocess(result)])
    if total == 1:
        print(json.dumps(result, indent=4))

print(f"{succ}/{total} = {succ/total*100}%")
