import os
import asyncio
import json
import random
from env import Actor2 as act
from typing import List, Dict, Any

actor = act()

task_type = [0]

succ, total = 0, 1000

for i in range(total):
    # id = random.randint(0, 1)
    id = 0
    task_id = 0
    task_id = task_type[id] * 100000000 + random.randint(0, 100000000)
    print(f"{'='*10} {task_id} {'='*10}")
    result = asyncio.run(actor.evaluate(task_id=task_id))

    print(f"{result['task_name'].split(":")[1]}:{task_id}:{result['score']}")
    succ += result['score']
    if total == 1:
        print(json.dumps(result, indent=4))

print(f"{succ}/{total} = {succ/total*100}%")
