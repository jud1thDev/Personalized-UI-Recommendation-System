import math
from collections import Counter
from typing import List


def entropy(sequence: List[str]) -> float:
    if not sequence:
        return 0.0
    cnt = Counter(sequence)
    total = sum(cnt.values())
    ent = 0.0
    for c in cnt.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent 