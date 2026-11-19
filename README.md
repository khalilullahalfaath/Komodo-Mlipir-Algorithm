# Komodo-Mlipir-Algorithm

## REQUIREMENTS

* use Python minimum of 3.10

## INSTALL & USAGE

```bash
pip install -e .          # or: pip install git+https://github.com/khalilullahalfaath/Komodo-Mlipir-Algorithm.git
python -m kma 1 50        # sanity run: function F1, dimension 50
```

```python
from kma import KMA
r = KMA(function_id=10, dimension=30, seed=0).run()   # dict: best_individual, opt_val, num_eva, fopt, ...
```

This package is the single source of truth for the algorithm (verified against the MATLAB original, see
`comparison/`). Variants such as the binary / feature-selection version subclass `KMA` and override
`_setup_problem`, `fitness` and `trim` instead of copying the algorithm. A different search space
(e.g. bit vectors) also overrides the operators (`generate_population`, `_crossover`, `_mutation`,
`_add_individual`, `_reposition`) and the movement steps (`_attract`, `_distract`, `_mlipir`, `_sexual`),
keeping the movement loops and `run()` here.
