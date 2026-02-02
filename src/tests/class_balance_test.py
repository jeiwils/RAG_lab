"""Quick check for label balance in sampled training examples.

This script counts labels after `build_training_examples` applies its sampling
scheme (hard negatives + random negatives). It does not measure the raw
supporting/non-supporting ratio in the original dataset.
"""

from collections import Counter
from src.utils.dataset_utils import load_dataset_split, build_training_examples


# DATASET = # dataset name, e.g. "hotpotqa"
# SPLIT = # split name, e.g. "train" or "dev"


train = load_dataset_split("train", dataset="hotpotqa")
examples = build_training_examples(train, hard_negatives=4, random_negatives=4)
cnt = Counter(int(ex.get("label", 0)) for ex in examples)
total = sum(cnt.values())
pos = cnt.get(1, 0)
neg = cnt.get(0, 0)
print("counts:", cnt)
print("pos_rate:", pos / total if total else 0)
print("neg:pos:", f"{neg}:{pos}" if pos else "inf")



