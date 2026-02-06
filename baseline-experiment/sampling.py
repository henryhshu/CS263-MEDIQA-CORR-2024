import json
import random
from pathlib import Path
from datasets import load_dataset

def save_indices(indices, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"indices": indices}, indent=2), encoding="utf-8")

def load_indices(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))["indices"]

dataset_test = load_dataset("mkieffer/MEDEC", split="test")
n = len(dataset_test)
print("Total test examples:", n)

sample_size = 50
seed = 203

rng = random.Random(seed)
sampled_indices = rng.sample(range(n), k=min(sample_size, n))
sampled_indices.sort()
save_indices(sampled_indices, "sampled_test_indices.json")
print("Saved indices.")

loaded_indices = load_indices("sampled_test_indices.json")
subset = dataset_test.select(loaded_indices)
print("Subset length:", len(subset))
