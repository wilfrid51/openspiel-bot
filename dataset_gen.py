import os
import asyncio
import json
import random
from env import Actor as act
from typing import List, Dict, Any

def preprocess(result):
    data = result['extra']
    data['reward'] = result['score']
    return data

def read_jsonl(path: str):
    """
    Read a JSONL file (one JSON object per line) into a list of dicts.
    """
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                raise ValueError(f"Expected a JSON object on line {line_no} of {path}, got {type(obj).__name__}")
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    """
    Write a list of dictionaries to a JSONL file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

actor = act()

def analyze_data():
    with open("merged.json", "r") as f:
        data = json.load(f)
    data.sort(key=lambda item: item['extra'].get('block_number', 0))
    data_per_game = {}
    for item in data:
        extra = item['extra']
        game_name = extra['game_name']
        task_id = extra['task_id']
        seed = extra['seed']
        if game_name not in data_per_game:
            data_per_game[game_name] = []
        data_per_game[game_name].append({
            "task_id": task_id,
            "seed": seed,
        })
    return data_per_game

already_data = read_jsonl("othello.jsonl")
_task_ids = [x['extra']['task_id'] for x in already_data]
# data_per_game = analyze_data()
for item in analyze_data()['othello']:
    task_id, seed = item['task_id'], item['seed']
    if task_id in _task_ids:
        continue
    print(f"{task_id}: {seed}")
    result = asyncio.run(actor.evaluate(task_id=task_id, seed=seed))
    print(result)
    write_jsonl(f"othello.jsonl", [result])
