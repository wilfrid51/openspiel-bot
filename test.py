import os
import sys
import asyncio
import json
import random
from env import Actor as act
from typing import List, Dict, Any

from datasets import load_dataset

actor = act()

task_type = [0]

task_seed = set()

# ds = load_dataset("top-50000/fit", split="train")

# def analyze_data():
#     with open("merged.json", "r") as f:
#         data = json.load(f)
#     data.sort(key=lambda item: item['extra'].get('block_number', 0))
#     data_per_game = {}
#     for item in data:
#         extra = item['extra']
#         game_name = extra['game_name']
#         task_id = extra['task_id']
#         seed = extra['seed']
#         if game_name not in data_per_game:
#             data_per_game[game_name] = []
#         data_per_game[game_name].append({
#             "task_id": int(task_id),
#             "seed": int(seed),
#         })
#     return data_per_game

# data_per_game = analyze_data()
# for item in analyze_data()['othello']:
#     task_id, seed = item['task_id'], item['seed']
#     print(f"{task_id}: {seed}")
#     result = asyncio.run(actor.evaluate(task_id=task_id, seed=seed))
#     print(result)

# for item in ds:
#     task_id, seed = item['task_id'], item['seed']
#     task_seed.add((task_id, seed))

succ, total = 0, 10

# for item in analyze_data()['goofspiel']:
# for item in task_seed:
for i in range(total):
    # task_id, seed = item['task_id'], item['seed']
    task_type = 4
    task_id = random.randint(task_type * 100000000, (task_type + 1) * 100000000)
    seed = random.randint(0, 50000000) * 2
    print(task_id, seed, flush=True)
    print(f"{'='*10} {task_id} {seed} {'='*10}", flush=True)
    result = asyncio.run(actor.evaluate(
        task_id=task_id,
        seed=seed,
        api_key="fdkfjsadfjdaf",
        model="testing_model",
        base_url="http://64.247.196.85:20000/v1",
        verbose=True,
    ))

    print(f"{result['task_name'].split(":")[1]}:{task_id}:{result['score']}", flush=True)
    succ += result['score']
    # total += 1
    # if total == 1:
    print(result, flush=True)

print(f"{succ}/{total} = {succ/total*100}%", flush=True)
