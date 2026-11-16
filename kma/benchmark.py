"""Port of Evaluation.m: the 23 classic benchmark functions F1-F23 (minimisation)."""

import numpy as np

_SHEKEL_A = np.array(
    [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
     [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]]
)
_SHEKEL_C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])

_H3_A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
_H3_P = np.array(
    [[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747],
     [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]]
)
_H6_A = np.array(
    [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
     [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
)
_H6_P = np.array(
    [[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
     [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
     [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
     [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]]
)
_H_C = np.array([1, 1.2, 3, 3.2])

_FOX = np.array([-32, -16, 0, 16, 32])
_FOX_A = np.vstack([np.tile(_FOX, 5), np.repeat(_FOX, 5)])  # 2 x 25

_KOW_A = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
_KOW_B = 1 / np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])


def _penalty(x, a, k=100, m=4):
    return np.sum(k * (x - a) ** m * (x > a) + k * (-x - a) ** m * (x < -a))


def evaluate(function_id, x):
    x = np.asarray(x, dtype=float)
    d = x.size
    i = np.arange(1, d + 1)

    match function_id:
        case 1:
            return np.sum(x**2)
        case 2:
            return np.sum(np.abs(x)) + np.prod(np.abs(x))
        case 3:
            return np.sum(np.cumsum(x) ** 2)
        case 4:
            return np.max(np.abs(x))
        case 5:
            return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
        case 6:
            return np.sum(np.floor(x + 0.5) ** 2)
        case 7:
            return np.sum(i * x**4) + np.random.random()
        case 8:
            return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
        case 9:
            return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * d
        case 10:
            return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
                    - np.exp(np.sum(np.cos(2 * np.pi * x)) / d) + 20 + np.e)
        case 11:
            return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1
        case 12:
            core = (10 * np.sin(np.pi * (1 + (x[0] + 1) / 4)) ** 2
                    + np.sum(((x[:-1] + 1) / 4) ** 2
                             * (1 + 10 * np.sin(np.pi * (1 + (x[1:] + 1) / 4)) ** 2))
                    + ((x[-1] + 1) / 4) ** 2)
            return (np.pi / d) * core + _penalty(x, 10)
        case 13:
            core = (np.sin(3 * np.pi * x[0]) ** 2
                    + np.sum((x[:-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1:]) ** 2))
                    + (x[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[-1]) ** 2))
            return 0.1 * core + _penalty(x, 5)
        case 14:  # Foxholes (labelled 16 in the original Evaluation.m, a typo)
            b = np.sum((x[:, None] - _FOX_A) ** 6, axis=0)
            return 1 / (1 / 500 + np.sum(1 / (np.arange(1, 26) + b)))
        case 15:  # Kowalik
            return np.sum(
                (_KOW_A - x[0] * (_KOW_B**2 + x[1] * _KOW_B)
                 / (_KOW_B**2 + x[2] * _KOW_B + x[3])) ** 2
            )
        case 16:  # Six Hump Camel
            return (4 * x[0] ** 2 - 2.1 * x[0] ** 4 + x[0] ** 6 / 3
                    + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4)
        case 17:  # Branin
            return ((x[1] - x[0] ** 2 * 5.1 / (4 * np.pi**2) + 5 / np.pi * x[0] - 6) ** 2
                    + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)
        case 18:  # Goldstein-Price
            x0, x1 = x[0], x[1]
            return ((1 + (x0 + x1 + 1) ** 2
                     * (19 - 14 * x0 + 3 * x0**2 - 14 * x1 + 6 * x0 * x1 + 3 * x1**2))
                    * (30 + (2 * x0 - 3 * x1) ** 2
                       * (18 - 32 * x0 + 12 * x0**2 + 48 * x1 - 36 * x0 * x1 + 27 * x1**2)))
        case 19:  # Hartman 3
            return -np.sum(_H_C * np.exp(-np.sum(_H3_A * (x - _H3_P) ** 2, axis=1)))
        case 20:  # Hartman 6
            return -np.sum(_H_C * np.exp(-np.sum(_H6_A * (x - _H6_P) ** 2, axis=1)))
        case 21 | 22 | 23:  # Shekel 5 / 7 / 10
            m = {21: 5, 22: 7, 23: 10}[function_id]
            return -np.sum(1 / (np.sum((x - _SHEKEL_A[:m]) ** 2, axis=1) + _SHEKEL_C[:m]))
    raise ValueError(f"unknown function_id {function_id}")
