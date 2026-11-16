"""python -m kma [function_id] [dimension]  -> quick sanity run on a benchmark function."""
import sys

from . import KMA

fid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
dim = int(sys.argv[2]) if len(sys.argv) > 2 else 50
kma = KMA(function_id=fid, dimension=dim, seed=0)
r = kma.run()
print(f"F{fid} dim {kma.var_num}: evaluations {r['num_eva']}, time {r['proc_time']:.2f}s, "
      f"optimum {kma.optimum_value}, found {r['opt_val']:.10g}")
