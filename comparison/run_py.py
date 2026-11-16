"""30 Python runs (seeds 1..30) of F1-F23, dimension 50 -> results/python.csv."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kma import KMA

res = []
for fid in range(1, 24):
    for r in range(1, 31):
        o = KMA(function_id=fid, dimension=50, seed=r).run()
        res.append([fid, r, o["opt_val"], o["num_eva"], o["proc_time"]])
    print("F", fid, flush=True)
np.savetxt(Path(__file__).parent / "results" / "python.csv", np.array(res), delimiter=",", fmt="%.17g")
