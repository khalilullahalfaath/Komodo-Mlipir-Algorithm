import numpy as np
from typing import Tuple, List
from algorithms.benchmarks import Benchmark


class KMA:
    def __init__(
        self, function_id: int, dimension: int, max_num_eva: int, pop_size: int
    ):
        self.function_id = function_id  # identity of the benchmark function
        self.dimension = (
            dimension  # dimension can be scaled up to thousands for the functions
        )
        self.max_num_eva = max_num_eva  # maximum number of evaluations
        self.pop_size = pop_size  # population size (number of komodo individuals)
        self.min_ada_pop_size = pop_size * 4  # minimum adaptive size
        self.max_ada_pop_size = pop_size * 40  # maximum adaptive size

        # get a bechmark function
        self.nvar, self.ub, self.lb, self.fthreshold_fx = Benchmark.get_function(
            self.function_id
        )
        self.ra = np.ones((1, self.nvar)) * self.ub
        self.rb = np.ones((1, self.nvar)) * self.lb

        self.big_males = None
        self.big_males_fx = None
        self.female = None
        self.female_fx = None
        self.small_males = None
        self.small_males_fx = None
        self.all_hq = None
        self.all_hq_fx = None
        self.mlipir_rate = (self.nvar - 1) / self.nvar
        self.mut_rate = 0.5
        self.mut_radius = 0.5

        self.pop = None
        self.fx = None

    def evaluation(self, x: np.ndarray) -> float:
        """
        Evaluate one komodo individual for the given function id

        Args:
            x: array of Komodo Individuals with shape ((n_var))

        Returns:
            Returns evaluation result for the komodo individuals

        Raises:
            ValueError: function_id is not defined
            ArrayOutOfBoundError:
            DivideByZero
        """
        return Benchmark.evaluate(x, self.function_id)

    def pop_cons_initialization(self, ps: int) -> np.ndarray:
        """
        Create a population constrained to the defined upper (ub) and lower (lb) bounds

        Args:
            ps: population size

        Returns:
            Returns a population constrained to the defined ub and lb with shape ((ps))

        Raises:


        """
        f1 = np.array([0.01, 0.01, 0.99, 0.99])
        f2 = np.array([0.01, 0.99, 0.01, 0.99])

        x = np.zeros((ps, self.nvar))

        individu_X = 0

        for nn in range(0, ps, 4):
            if ps - nn >= 4:
                # n_loc = number of locations (at the four corners of the problem landscape)
                n_loc = 4
            else:
                n_loc = ps - nn

            ss = 0

            while ss < n_loc:
                temp = np.zeros((1, self.nvar))
                for i in range(self.nvar // 2):
                    temp[:, [i]] = self.rb[:, i] + (self.ra[:, i] - self.rb[:, i]) * (
                        f1[ss,] + ((np.random.rand() * 2) - 1) * 0.01
                    )

                for i in range(self.nvar // 2, self.nvar):
                    temp[:, [i]] = self.rb[:, i] + (self.ra[:, i] - self.rb[:, i]) * (
                        f2[ss,] + ((np.random.rand() * 2) - 1) * 0.01
                    )

                x[[individu_X], :] = temp
                individu_X += 1
                ss += 1

        return x

    def move_big_males_female_first_stage(self):
        # Implement MoveBigMalesFemaleFirstStage logic here
        pass

    def move_big_males_female_second_stage(self):
        # Implement MoveBigMalesFemaleSecondStage logic here
        pass

    def move_small_males_first_stage(self):
        # Implement MoveSmallMalesFirstStage logic here
        pass

    def move_small_males_second_stage(self):
        # Implement MoveSmallMalesSecondStage logic here
        pass

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        # Implement CrossOver logic here
        pass

    def mutation(self) -> np.ndarray:
        # Implement Mutation logic here
        pass

    def adding_pop(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        # Implement AddingPop logic here
        pass

    def reposition(self, x: np.ndarray, fx: float) -> Tuple[np.ndarray, float]:
        # Implement Reposition logic here
        pass

    def replacement(
        self, x: np.ndarray, fx: np.ndarray, y: np.ndarray, fy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Implement Replacement logic here
        pass

    def trimr(self, x: np.ndarray) -> np.ndarray:
        # Implement trimr logic here
        pass

    def run(
        self,
    ) -> Tuple[np.ndarray, float, int, List[float], List[float], float, List[int]]:

        # generate the initial population of population size komodo individuals
        self.pop = self.pop_cons_initialization(self.pop_size)

        # calculate all the fitness values of the population size komodo individuals
        self.fx = np.zeros((1, self.pop_size))

        for i in range(self.pop_size):
            self.fx[:, [i]] = self.evaluation(self.pop[[i], :])

        print(self.fx)

        best_indiv = np.zeros(self.dimension)
        opt_val = 0.0
        num_eva = self.max_num_eva
        fopt = [0.0] * num_eva
        fmean = [0.0] * num_eva
        proc_time = 0.0
        evo_pop_size = [self.pop_size] * num_eva
        return best_indiv, opt_val, num_eva, fopt, fmean, proc_time, evo_pop_size
