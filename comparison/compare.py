"""Print the comparison tables from comparison/results/*.csv (needs scipy)."""
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).resolve().parent / "results"
L = lambda n: np.loadtxt(R / n, delimiter=",")

# 1. benchmark functions on identical points
A = L("eval_points.csv")  # fid, python value, matlab value
rel = np.abs(A[:, 1] - A[:, 2]) / np.maximum(1e-12, np.abs(A[:, 2]))
print(f"Benchmark values, {len(A)} points: max relative diff = {rel.max():.2e}\n")

# 2. KMA runs
mo, mf, py = L("matlab_orig.csv"), L("matlab_fix.csv"), L("python.csv")
g = lambda a, f: a[a[:, 0] == f]
fmt = lambda a: f"{a[:, 2].mean():.3e} +- {a[:, 2].std():.1e}"


def pval(a, b):
    if a[:, 2].std() == 0 and b[:, 2].std() == 0 and a[0, 2] == b[0, 2]:
        return 1.0
    return stats.mannwhitneyu(a[:, 2], b[:, 2]).pvalue


print(f"{'F':<4}{'MATLAB orig':>24}{'MATLAB fix':>24}{'Python':>24}  p(orig,py) p(fix,py)  evals o/f/p")
for f in range(1, 24):
    o, x, p = g(mo, f), g(mf, f), g(py, f)
    print(f"F{f:<3}{fmt(o):>24}{fmt(x):>24}{fmt(p):>24}  {pval(o, p):.3f}      {pval(x, p):.3f}"
          f"      {o[:, 3].mean():.0f}/{x[:, 3].mean():.0f}/{p[:, 3].mean():.0f}")
print(f"\nmean time/run (s): orig {mo[:, 4].mean():.2f}  fix {mf[:, 4].mean():.2f}  python {py[:, 4].mean():.2f}")
