import numpy as np

from benchmark import evaluate
from get_function import get_function
from kma import KMA


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
