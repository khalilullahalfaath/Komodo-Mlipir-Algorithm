from pathlib import Path

import numpy as np
import scipy.io as sio

from kma import KMA, evaluate
from kma import get_function


def test_known_optima():
    # (function_id, optimum point, expected value)
    cases = [
        (1, np.zeros(5), 0), (2, np.zeros(5), 0), (3, np.zeros(5), 0),
        (4, np.zeros(5), 0), (5, np.ones(5), 0), (6, np.zeros(5), 0),
        (8, np.full(5, 420.9687), -418.9829 * 5), (9, np.zeros(5), 0),
        (10, np.zeros(5), 0), (11, np.zeros(5), 0), (12, np.full(5, -1.0), 0),
        (13, np.ones(5), 0), (14, np.array([-32.0, -32.0]), 0.998),
        (16, np.array([0.08983, -0.7126]), -1.0316),
        (17, np.array([np.pi, 2.275]), 0.398), (18, np.array([0.0, -1.0]), 3),
        (19, np.array([0.114614, 0.555649, 0.852547]), -3.86),
        (21, np.full(4, 4.0), -10.1532), (22, np.full(4, 4.0), -10.4029),
        (23, np.full(4, 4.0), -10.5364),
    ]
    for fid, x, expected in cases:
        assert abs(evaluate(fid, x) - expected) < 5e-2, (fid, evaluate(fid, x))


def test_all_functions_run():
    for fid in range(1, 24):
        _, _, _, n = get_function(fid, 10)
        assert np.isfinite(evaluate(fid, np.full(n, 0.5)))


def test_kma_sphere():
    r = KMA(function_id=1, dimension=10, max_eval_number=5000, seed=0).run()
    assert r["opt_val"] < 1e-3
    assert r["num_eva"] <= 5000 + 200


def test_kma_fixed_dim_and_stage2():
    r = KMA(function_id=14, seed=0).run()  # fixed 2-D, multimodal
    assert r["opt_val"] < 5
    r = KMA(function_id=10, dimension=10, seed=0).run()
    assert r["evo_pop_size"].max() > 5  # reached the self-adaptive stage


def test_initial_population_matches_matlab():
    # pop.mat: initial population of the MATLAB code for F5 (bounds +-30), dimension 50
    pop = KMA(function_id=5, dimension=50).generate_population(5)
    matlab = sio.loadmat(Path(__file__).parent / "data" / "pop.mat")["Pop"]
    assert pop.shape == matlab.shape
    np.testing.assert_allclose(np.sort(pop.ravel())[::50], np.sort(matlab.ravel())[::50], atol=1.0)


def test_initial_fitness_matches_matlab():
    # fitness of an F1 initial population (bounds +-100) from a MATLAB run
    kma = KMA(function_id=1, dimension=50)
    matlab_fx = np.array([4.788615799813203e05, 4.795563179041033e05, 4.819387031017045e05,
                          4.795778390078847e05, 4.816324607430795e05])
    fx = np.sort([kma.fitness(p) for p in kma.generate_population(5)])
    np.testing.assert_allclose(fx, np.sort(matlab_fx), rtol=0.01)


def test_hooks_allow_subclass():
    class Shifted(KMA):  # a different problem through the hooks only
        def _setup_problem(self):
            self.var_num, self.optimum_value = 4, 0.0
            self.lower_bound, self.upper_bound = np.full(4, -5.0), np.full(4, 5.0)

        def fitness(self, x):
            return float(np.sum((x - 1.0) ** 2))

    r = Shifted(max_eval_number=3000, seed=0).run()
    assert r["opt_val"] < 1e-3 and np.allclose(r["best_individual"], 1.0, atol=1e-2)


def test_fitness_hook_sees_progress():
    seen = []

    class Probe(KMA):
        def fitness(self, x):
            seen.append(self.num_eva / self.max_eval_number)
            return super().fitness(x)

    Probe(function_id=10, dimension=10, max_eval_number=3000, seed=0).run()
    assert seen[0] == 0  # initial population is evaluated before any counting
    assert all(a <= b for a, b in zip(seen, seen[1:]))  # progress never goes backwards
    assert 0.9 < max(seen) <= 1.01  # and reaches the end of the budget
