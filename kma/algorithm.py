"""Komodo Mlipir Algorithm (Suyanto et al., 2021), port of KMA2D.m without the plotting."""

import math
import time

import numpy as np

from .benchmark import evaluate
from .get_function import get_function


def levy(n, m, beta):
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.normal(0, sigma, (n, m))
    v = np.random.normal(0, 1, (n, m))
    return u / np.abs(v) ** (1 / beta)


class KMA:
    """KMA for minimisation. Subclasses (e.g. a binary / feature-selection variant) override
    `_setup_problem`, `fitness` and `trim`; everything else is shared. Attributes a subclass
    needs inside those hooks must be set before calling `super().__init__`."""

    mut_rate = 0.5  # probability of the female mutation
    mut_radius = 0.5  # radius limiting the female mutation step

    def __init__(
        self,
        function_id=1,
        dimension=50,
        max_eval_number=25000,
        population_size=5,
        min_adaptive_size=4,  # min adaptive population = population_size * this
        max_adaptive_size=40,  # max adaptive population = population_size * this
        max_gen_exam1=100,  # first stage: generation at which the improve rate is checked
        max_gen_exam2=1000,  # first stage: maximum number of generations
        seed=None,
    ) -> None:
        self.function_id = function_id
        self.dimension = dimension
        self.max_eval_number = max_eval_number
        self.pop_size = population_size
        self.min_adaptive_pop_size = population_size * min_adaptive_size
        self.max_adaptive_pop_size = population_size * max_adaptive_size
        self.max_gen_exam1 = max_gen_exam1
        self.max_gen_exam2 = max_gen_exam2
        if seed is not None:
            np.random.seed(seed)
        self._setup_problem()

    # ---------- hooks ----------
    def _setup_problem(self):
        """Set var_num, lower_bound, upper_bound and optimum_value (the stop threshold)."""
        ub, lb, self.optimum_value, self.var_num = get_function(
            self.function_id, self.dimension
        )
        self.upper_bound = np.full(self.var_num, float(ub))
        self.lower_bound = np.full(self.var_num, float(lb))

    def fitness(self, x):
        return evaluate(self.function_id, x)

    def trim(self, x):
        return np.clip(x, self.lower_bound, self.upper_bound)

    def generate_population(self, size):
        """Constrained initialisation: individuals near the four corners of the landscape."""
        f1 = np.array([0.01, 0.01, 0.99, 0.99])
        f2 = np.array([0.01, 0.99, 0.01, 0.99])
        half = self.var_num // 2
        span = self.upper_bound - self.lower_bound
        pop = np.zeros((size, self.var_num))
        for k in range(size):
            f = np.where(np.arange(self.var_num) < half, f1[k % 4], f2[k % 4])
            pop[k] = self.lower_bound + span * (f + (np.random.rand(self.var_num) * 2 - 1) * 0.01)
        return pop

    def _evaluate_all(self, pop):
        return np.array([self.fitness(p) for p in pop])

    def _replacement(self, x, fx, y, fy):
        """Keep the best len(x) individuals out of old (x) and new (y)."""
        xy, fxy = np.vstack([x, y]), np.concatenate([fx, fy])
        idx = np.argsort(fxy, kind="stable")[: len(x)]
        return xy[idx], fxy[idx]

    def _crossover(self, p1, p2):
        r = np.random.rand(self.var_num)
        return self.trim(r * p1 + (1 - r) * p2), self.trim(r * p2 + (1 - r) * p1)

    def _mutation(self, female):
        step = self.mut_radius * (self.upper_bound - self.lower_bound)
        mask = np.random.rand(self.var_num) < self.mut_rate
        return self.trim(female + mask * (2 * np.random.rand(self.var_num) - 1) * step)

    def _add_individual(self, x):
        new = self.trim(x + 0.05 * levy(1, self.var_num, 1.5)[0] * np.abs(self.upper_bound - self.lower_bound))
        return new, self.fitness(new)

    def _reposition(self, x, fx):
        step = self.mut_radius * (self.upper_bound - self.lower_bound)
        mask = np.random.rand(self.var_num) < self.mut_rate
        new = self.trim(x + mask * (2 * np.random.rand(self.var_num) - 1) * self.mut_radius * step)
        new_fx = self.fitness(new)
        return (new, new_fx) if new_fx < fx else (x, fx)

    # ---------- movements ----------
    def _move_big_males_female(self, bm, bm_fx, female, female_fx, hq, hq_fx, max_fol):
        """Big males follow other high-quality males; the winner then mates the female."""
        temp, temp_fx = bm.copy(), bm_fx.copy()
        for s in range(len(temp)):
            vm = np.zeros(self.var_num)
            limit = np.random.randint(1, max_fol + 1)
            for fol, ind in enumerate(np.random.permutation(len(hq)), 1):
                if ind != s:
                    if hq_fx[ind] < temp_fx[s] or np.random.rand() < 0.5:  # attraction
                        vm += np.random.rand() * (hq[ind] - temp[s])
                    else:  # distraction
                        vm += np.random.rand() * (temp[s] - hq[ind])
                if fol >= limit:
                    break
            temp[s] = self.trim(temp[s] + vm)
            temp_fx[s] = self.fitness(temp[s])
        bm, bm_fx = self._replacement(bm, bm_fx, temp, temp_fx)

        if bm_fx[0] < female_fx or np.random.rand() < 0.5:  # sexual reproduction
            o1, o2 = self._crossover(bm[0], female)
            fx1, fx2 = self.fitness(o1), self.fitness(o2)
            child, child_fx = (o1, fx1) if fx1 < fx2 else (o2, fx2)
        else:  # asexual reproduction
            child = self._mutation(female)
            child_fx = self.fitness(child)
        if child_fx < female_fx:
            female, female_fx = child, child_fx
        return bm, bm_fx, female, female_fx

    def _move_small_males(self, sm, hq, mlipir_rate, max_fol):
        """Small males 'mlipir': move along a random subset of dimensions towards big males."""
        new_sm = sm.copy()
        d = min(max(int(math.floor(mlipir_rate * self.var_num + 0.5)), 1), self.var_num - 1)
        for w in range(len(sm)):
            v = np.zeros(self.var_num)
            limit = np.random.randint(1, max_fol + 1)
            for fol, ind in enumerate(np.random.permutation(len(hq)), 1):
                b = np.zeros(self.var_num)
                b[np.random.permutation(self.var_num)[:d]] = 1
                v = v + np.random.rand(self.var_num) * (hq[ind] * b) - sm[w] * b
                if fol >= limit:
                    break
            new_sm[w] = self.trim(sm[w] + v)
        return new_sm, self._evaluate_all(new_sm)

    # ---------- main loop ----------
    def run(self):
        start = time.time()
        n, thr = self.pop_size, self.optimum_value

        pop = self.generate_population(n)
        fx = self._evaluate_all(pop)
        idx = np.argsort(fx, kind="stable")
        pop, fx = pop[idx], fx[idx]
        one_elit = fx[0]
        nbm = n // 2

        # ---- first stage: is the function simple or complex? ----
        mlipir_rate = (self.var_num - 1) / self.var_num
        is_global, improve_rate, num_eva, gen, gen_improve = False, 0.0, 0, 0, 0
        fopt, fmean, evo_pop_size = [], [], []

        while gen < self.max_gen_exam2:
            gen += 1
            num_eva += n
            bm, bm_fx = pop[:nbm], fx[:nbm]
            female, female_fx = pop[nbm], fx[nbm]
            sm = pop[nbm + 1:]

            bm, bm_fx, female, female_fx = self._move_big_males_female(
                bm, bm_fx, female, female_fx, bm, bm_fx, max_fol=2
            )
            sm, sm_fx = self._move_small_males(sm, bm, mlipir_rate, max_fol=1)

            pop = np.vstack([bm, female, sm])
            fx = np.concatenate([bm_fx, [female_fx], sm_fx])
            idx = np.argsort(fx, kind="stable")
            pop, fx = pop[idx], fx[idx]
            best, opt_val = pop[0], fx[0]

            fopt.append(opt_val)
            fmean.append(fx.mean())
            evo_pop_size.append(n)

            if opt_val < one_elit:
                gen_improve += 1
                improve_rate = gen_improve / gen
                one_elit = opt_val
            if opt_val <= thr:
                is_global = True
                break
            if gen == self.max_gen_exam1 and improve_rate < 0.5:
                break

        # ---- second stage: self-adaptive population size ----
        if not is_global and num_eva <= self.max_eval_number:
            swarm = len(pop)
            nbm = swarm // 2
            step = swarm  # increment / decrement of the adaptive population size
            max_ada = self.max_adaptive_pop_size
            max_improve = max_stagnan = 2
            gen_improve = gen_stagnan = 0

            cons = self.generate_population(max_ada - swarm)
            pop = np.vstack([pop, cons])
            fx = np.concatenate([fx, self._evaluate_all(cons)])
            one_elit = fx.min()

            while num_eva < self.max_eval_number:
                ada = len(pop)
                n_swarms = ada // swarm

                # collect the big males of every micro-swarm (row k*nbm; the MATLAB original
                # indexes with `ind`, which pads AllHQ with zero rows)
                all_hq = np.zeros((n_swarms * nbm, self.var_num))
                all_hq_fx = np.zeros(n_swarms * nbm)
                for k in range(n_swarms):
                    sl = slice(k * swarm, (k + 1) * swarm)
                    order = np.argsort(fx[sl], kind="stable")[:nbm]
                    all_hq[k * nbm:(k + 1) * nbm] = pop[sl][order]
                    all_hq_fx[k * nbm:(k + 1) * nbm] = fx[sl][order]

                for k in range(n_swarms):
                    sl = slice(k * swarm, (k + 1) * swarm)
                    order = np.argsort(fx[sl], kind="stable")
                    ms, ms_fx = pop[sl][order], fx[sl][order]
                    bm, bm_fx = ms[:nbm], ms_fx[:nbm]
                    female, female_fx = ms[nbm], ms_fx[nbm]
                    sm = ms[nbm + 1:]

                    g_hq = np.vstack([bm, all_hq])
                    g_hq_fx = np.concatenate([bm_fx, all_hq_fx])
                    bm, bm_fx, female, female_fx = self._move_big_males_female(
                        bm, bm_fx, female, female_fx, g_hq, g_hq_fx, max_fol=3
                    )
                    sm, sm_fx = self._move_small_males(sm, g_hq, 0.5, max_fol=3)

                    all_hq[k * nbm:(k + 1) * nbm] = bm
                    all_hq_fx[k * nbm:(k + 1) * nbm] = bm_fx
                    pop[sl] = np.vstack([bm, female, sm])
                    fx[sl] = np.concatenate([bm_fx, [female_fx], sm_fx])
                    num_eva += swarm
                    if fx.min() <= thr:
                        break

                perm = np.random.permutation(len(fx))
                pop, fx = pop[perm], fx[perm]
                i_min = fx.argmin()
                best, opt_val = pop[i_min], fx[i_min]
                fopt.append(opt_val)
                fmean.append(fx.mean())
                if opt_val <= thr:
                    break

                # self-adaptation of the population size
                if opt_val < one_elit:
                    gen_improve += 1
                    gen_stagnan = 0
                    one_elit = opt_val
                else:
                    gen_stagnan += 1
                    gen_improve = 0

                if gen_improve > max_improve:  # shrink: keep the best individuals
                    ada = max(ada - step, self.min_adaptive_pop_size)
                    keep = np.argsort(fx, kind="stable")[:ada]
                    pop, fx = pop[keep], fx[keep]
                    gen_improve = 0

                if gen_stagnan > max_stagnan:  # grow, or reposition when already at max
                    old = len(pop)
                    ada = old + step
                    if ada > max_ada:
                        ada = old
                    if ada > old:
                        added = [self._add_individual(best) for _ in range(ada - old)]
                        pop = np.vstack([pop, [a[0] for a in added]])
                        fx = np.concatenate([fx, [a[1] for a in added]])
                        num_eva += ada - old
                    else:
                        for j in range(len(pop)):
                            pop[j], fx[j] = self._reposition(pop[j], fx[j])
                        num_eva += len(pop)
                    gen_stagnan = 0

                perm = np.random.permutation(len(fx))
                pop, fx = pop[perm], fx[perm]
                evo_pop_size.append(ada)
                gen += 1

        return {
            "best_individual": best,
            "opt_val": opt_val,
            "num_eva": num_eva,
            "fopt": np.array(fopt),
            "fmean": np.array(fmean),
            "proc_time": time.time() - start,
            "evo_pop_size": np.array(evo_pop_size),
        }

