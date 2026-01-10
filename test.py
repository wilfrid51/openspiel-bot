import os
import asyncio
import json
import random
from env import Actor as act
from typing import List, Dict, Any

from datasets import load_dataset

actor = act()

task_type = [0]

task_seed = set()

dataset = load_dataset("top-50000/liars_dice", split="train")
for ds in dataset:
    task_id = ds['task_id']
    seed = ds['seed']
    task_seed.add((task_id, seed))
    print(task_id, seed)


succ, total = 0, len(task_seed)

for (task_id, seed) in task_seed:
    # task_id = task_ids[i]
    print(f"{'='*10} {task_id} {seed} {'='*10}")
    result = asyncio.run(actor.evaluate(
        task_id=task_id,
        seed=seed,
        model="ATL-Machine/affine-game-test",
        # Port 20000 is typically served over plain HTTP unless you put a TLS reverse-proxy in front.
        base_url="http://64.247.196.85:20000/v1"
    ))

    print(f"{result['task_name'].split(":")[1]}:{task_id}:{result['score']}")
    succ += result['score']
    # if total == 1:
    print(result)

print(f"{succ}/{total} = {succ/total*100}%")
