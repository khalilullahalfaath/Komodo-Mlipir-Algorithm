"""Write random points + Python benchmark values to results/points.csv (columns: fid, n, py_value, x...)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark import evaluate
from get_function import get_function

rng = np.random.default_rng(0)
rows = []
for fid in range(1, 24):
    if fid == 7:  # contains rand
        continue
    ub, lb, _, n = get_function(fid, 50)
    for _ in range(5):
        x = rng.uniform(lb, ub, n)
        rows.append([fid, n, evaluate(fid, x)] + list(x) + [np.nan] * (50 - n))
np.savetxt(Path(__file__).parent / "results" / "points.csv", np.array(rows), delimiter=",", fmt="%.17g")
