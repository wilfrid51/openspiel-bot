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

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    """
    Write a list of dictionaries to a JSONL file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

actor = act()

with open("merged.json", "r") as f:
    data = json.load(f)

data_per_game = {}

# Sort items in `data` by the value of 'block_number' in the 'extra' field
data.sort(key=lambda item: item['extra'].get('block_number', 0))

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

# for game_name, data in data_per_game.items():
#     print(f"{game_name}: {(data)}")
#     for item in data:
#         print(f"\t{item['task_id']}: {item['seed']}")
#     print()

liars_dice = data_per_game['liars_dice']
leduc_poker = data_per_game['leduc_poker']
clobber = data_per_game['clobber']
othello = data_per_game['othello']
gin_rummy = data_per_game['gin_rummy']
goofspiel = data_per_game['goofspiel']
backgammon = data_per_game['backgammon']

for item in othello:
    task_id, seed = item['task_id'], item['seed']
    print(f"{task_id}: {seed}")
    result = asyncio.run(actor.evaluate(task_id=task_id, seed=seed))
    print(result)
