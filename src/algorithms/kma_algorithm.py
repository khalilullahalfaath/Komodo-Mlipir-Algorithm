import numpy as np
from numpy.random import default_rng
from typing import Tuple, List
from src.algorithms.benchmarks import Benchmark
import math


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
            self.dimension, self.function_id
        )
        self.ra = np.ones((1, self.nvar)) * self.ub
        self.rb = np.ones((1, self.nvar)) * self.lb

        self.num_BM = int(np.floor(self.pop_size / 2))

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

        self.one_elit_fx = None

        self.pop = None
        self.fx = None

    def evaluation(self, x: np.ndarray) -> float:
        """
        Evaluate one komodo individual for the given function id

        Args:
            x: array of Komodo Individuals with shape ((nvar))

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

        for nn in range(0, ps, 4):  # Adjust to start from 0 for Python
            if ps - nn >= 4:
                n_loc = 4
            else:
                n_loc = ps - nn

            ss = 0

            while ss < n_loc:
                temp = np.zeros((1, self.nvar))
                for i in range(0, math.floor(self.nvar / 2)):
                    temp[0, i] = self.rb[0, i] + (self.ra[0, i] - self.rb[0, i]) * (
                        f1[ss] + ((np.random.rand() * 2) - 1) * 0.01
                    )

                for i in range(math.floor(self.nvar / 2), self.nvar):
                    temp[0, i] = self.rb[0, i] + (self.ra[0, i] - self.rb[0, i]) * (
                        f2[ss] + ((np.random.rand() * 2) - 1) * 0.01
                    )

                x[individu_X, :] = temp
                individu_X += 1
                ss += 1

        return x

    def move_big_males_female_first_stage(self):
        """
        Move Big Males and Female in the first stage. The winner mates Female (if the Female wants)


        Args:


        Returns:


        Raises:
            ValueError: array shape is not correct

        """
        hq = np.copy(self.big_males)
        hq_fx = np.copy(self.big_males_fx)

        temp_small_males = np.copy(self.big_males)
        temp_small_males_fx = np.copy(self.big_males_fx)

        for ss in range(temp_small_males.shape[0]):
            max_fol_hq = np.random.randint(1, 3)
            vm = np.zeros((1, self.dimension))
            rhq = np.random.permutation(hq.shape[0])
            fol_hq = 0

            for fs in range(len(rhq)):
                ind = rhq[fs]
                if ind != ss:
                    # Semi randomly select sn individual to define an attraction or a distraction
                    if (hq_fx[:, ind] < temp_small_males_fx[:, ss]) or (
                        np.random.rand() < 0.5
                    ):
                        vm += np.random.rand() * (hq[ind, :] - temp_small_males[ss, :])
                    else:
                        vm += np.random.rand() * (temp_small_males[ss, :] - hq[ind, :])

                    fol_hq += 1
                    if fol_hq >= max_fol_hq:
                        break

            new_big_males = temp_small_males[ss, :].copy() + vm
            new_big_males = self.trimr(new_big_males)

            temp_small_males[ss, :] = new_big_males
            temp_small_males_fx[:, ss] = self.evaluation(new_big_males)

        self.big_males, self.big_males_fx = self.replacement(
            self.big_males, self.big_males_fx, temp_small_males, temp_small_males_fx
        )

        winner_big_males = self.big_males[0, :].copy().reshape(1, -1)
        winner_big_males_fx = self.big_males_fx[:, 0].copy().reshape(1, -1)

        if winner_big_males.shape != (1, self.nvar) or winner_big_males_fx.shape != (
            1,
            1,
        ):
            raise ValueError("Array big males is not correct")

        if (winner_big_males_fx < self.female_fx) or (np.random.rand() < 0.5):
            offsprings = self.crossover(winner_big_males, self.female)
            fx1 = self.evaluation(offsprings[0, :]).reshape(1, 1)
            fx2 = self.evaluation(offsprings[1, :]).reshape(1, 1)

            # keep the best position of female
            if fx1 < fx2:
                if fx1 < self.female_fx:
                    self.female = offsprings[0, :].copy()
                    self.female_fx = fx1.copy()
            else:
                if fx2 < self.female_fx:
                    self.female = offsprings[0, :].copy()
                    self.female_fx = fx2.copy()
        else:
            # asexual reproduction
            new_female = self.mutation()

            if new_female.shape() != (1, self.nvar):
                new_female = new_female.reshape(1, -1)

                if new_female.shape() != (1, self.nvar):
                    raise ValueError("Array female after mutation is not correct")

            fx = self.evaluation(new_female)

            if fx < self.female_fx:
                self.female = new_female.copy()
                self.female_fx = fx.copy()

    def move_big_males_female_second_stage(self):
        pass

    def move_small_males_first_stage(self):
        """
        Move (Mlipir) Small Males aside with the maximum Mlipir Rate to do a high-exploitative low-explorative searching

        Params:


        Returns:


        Raises:
            ValueError: array shape is not correct
        """
        hq = np.copy(self.big_males)
        temp_weak_males = np.copy(self.small_males)
        temp_weak_males_fx = np.copy(self.small_males_fx)
        max_fol_hq = 1

        for ww in range(self.small_males.shape[0]):
            vmlipir = np.zeros((1, self.nvar))
            rhq = np.random.permutation(hq.shape[0])
            fol_hq = 0

            for fs in range(len(rhq)):
                ind = rhq[fs].copy()
                A = np.random.permutation(self.nvar)
                D = int(np.round(self.mlipir_rate * self.nvar))

                if D >= self.nvar:
                    D = self.nvar - 1

                if D < 1:
                    D = 1

                M = A[0 : D - 1]  # Moving the WM based on the D-attributes
                B = np.zeros((1, self.nvar))  # Initialize Binary pattern
                B[0, M] = 1  # Binary pattern

                vmlipir += (np.random.rand(1, self.nvar) * (hq[ind, :] * B)) - (
                    self.small_males[ww, :] * B
                )

                fol_hq += 1
                if fol_hq >= max_fol_hq:
                    break

            new_small_males = self.small_males[ww, :].copy() + vmlipir
            new_small_males = self.trimr(new_small_males)
            temp_weak_males[ww, :] = new_small_males
            temp_weak_males_fx[:, ww] = self.evaluation(new_small_males)

        self.small_males = temp_weak_males
        self.small_males_fx = temp_weak_males_fx

    def move_small_males_second_stage(self):
        # Implement MoveSmallMalesSecondStage logic here
        pass

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Whole arithmetic crossover

        Params:
            parent1, parent2: mating individuals with the shape (1, nvar)

        Returns:
            Returns two individual offsprings that are resulted from two parents crossover with the shape of (1, nvar)

        Raises:

        """

        if parent1.shape != (1, self.nvar) or parent2.shape != (1, self.nvar):
            parent1 = parent1.reshape(1, -1)
            parent2 = parent2.reshape(1, -1)

            if parent1.shape != (1, self.nvar) or parent2.shape != (1, self.nvar):
                raise ValueError("Array parents shape are not correct")

        offsprings = np.zeros((2, self.nvar))

        for ii in range(self.nvar):
            r_val = np.random.rand()
            offsprings[0, ii] = (r_val * parent1[0, ii]) + (
                (1 - r_val) * parent2[0, ii]
            )
            offsprings[1, ii] = (r_val * parent2[0, ii]) + (
                (1 - r_val) * parent1[0, ii]
            )
        offsprings[0, :] = self.trimr(offsprings[0, :])
        offsprings[1, :] = self.trimr(offsprings[1, :])

        return offsprings

    def mutation(self) -> np.ndarray:
        """
        mutation of the only female


        Params:


        Returns:
            Returns a new female mutated individual with shape of (1, n_var) same with the original female


        """
        new_female = np.copy(self.female)  # Initialize a new Female

        if new_female.shape != (1, self.nvar):
            new_female = new_female.reshape(1, -1)
            if new_female.shape() != (1, self.nvar):
                raise ValueError("Array female after mutation is not correct")

        max_step = self.mut_radius * (
            self.ra - self.rb
        )  # Maximum step of the Female mutation

        for i in range(self.nvar):
            if (
                np.random.rand() < self.mut_rate
            ):  #  Check if a random value is lower than the Mutation Rate
                new_female[0, i] = (
                    self.female[0, i] + (2 * np.random.rand() - 1) * max_step[0, i]
                )

        new_female = self.trimr(
            new_female
        )  # Limit the values into the given dimensional boundaries
        return new_female

    def adding_pop(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        # Implement AddingPop logic here
        pass

    def reposition(self, x: np.ndarray, fx: float) -> Tuple[np.ndarray, float]:
        # Implement Reposition logic here
        pass

    def replacement(
        self, x: np.ndarray, fx: np.ndarray, y: np.ndarray, fy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Replacement: sort the old and new populations and select the best ones

        Parameters:
        X  : old population of LX individuals with the shape of (1,nvar)
        FX : old fitness with the shape of (1,nvar)
        Y  : new population with the shape of (1,nvar)
        FY : new fitness with the shape of (1,nvar)

        Returns:
        Z  : survivor of LX individuals with the shape of (1,nvar)
        FZ : survivor fitness with the shape of (1,nvar)

        Raises:

        """

        lx = x.shape[0]  # Number of individuals in old population
        xy = np.vstack((x, y))  # Joint individuals of old and new population
        fxfy = np.hstack((fx, fy))  # Joint fitness values of old and new population

        # Sort all fitness values and get sorted indices
        sorted_ind = np.argsort(fxfy[0])

        # Select the best individuals
        z = np.copy(xy[sorted_ind[:lx], :])
        fz = np.copy(fxfy[:, sorted_ind[:lx]])

        return z, fz

    def trimr(self, x: np.ndarray) -> np.ndarray:
        """
        Limit the values into the given dimensional boundaries

        Params:
            x: array shape (1,nvar) of Komodo individuals

        Returns:
            Returns new array of komodo individuals with shape (1,nvar) but have trimmed to dimensional boundaries

        Raises:

        """
        return np.clip(np.copy(x), self.rb, self.ra)

    def run(
        self,
    ) -> Tuple[np.ndarray, float, int, List[float], List[float], float, List[int]]:

        # generate the initial population of population size komodo individuals
        self.pop = self.pop_cons_initialization(self.pop_size)

        # calculate all the fitness values of the population size komodo individuals
        self.fx = np.zeros((1, self.pop_size))

        for i in range(self.pop_size):
            self.fx[:, [i]] = self.evaluation(self.pop[[i], :])

        # print(self.fx)

        # sort the fitness values of the initial population
        index_fx = np.argsort(self.fx[0])

        self.fx, self.pop = self.fx[:, index_fx], self.pop[index_fx, :]

        self.one_elit_fx = self.fx[0, 0]

        #
        #
        # First Stage: examining if the benchmark function is simple or complex
        #
        #

        is_global = 0
        # Boolean to check if the global optimum is reached
        improve_rate = 0
        # Improve rate to examine the benchmark function
        num_eva = 0
        # Number of evalutions
        gen = 0
        # Generation
        max_gen_exam1 = 100
        # Maximum generation of the first examination
        max_gen_exam2 = 1000
        # Maximum generation of the second examination

        fopt = []
        # Best-so-far fitness value in each generation
        fmean = []
        # Mean fitness value in each generation
        evo_pop_size = []
        # Population Size used in each generation
        gen_improve = 0
        # Generation counter to check the improvement condition

        while gen < max_gen_exam2:
            gen += 1
            num_eva += self.pop_size

            self.big_males = np.copy(self.pop[: self.num_BM, :])
            self.big_males_fx = np.copy(self.fx[:, : self.num_BM])

            self.female = np.copy(self.pop[self.num_BM, :]).reshape(1, -1)
            self.female_fx = np.copy(self.fx[:, self.num_BM]).reshape(1, -1)

            if self.female.shape != (1, self.nvar):
                raise ValueError("Female shape is incorrect")

            self.small_males = np.copy(self.pop[self.num_BM + 1 :, :])
            self.small_males_fx = np.copy(self.fx[:, self.num_BM + 1 :])

            # move big makes and female
            self.move_big_males_female_first_stage()
            # move small males
            self.move_small_males_first_stage()

            self.pop = np.vstack((self.big_males, self.female, self.small_males))
            self.fx = np.hstack(
                (self.big_males_fx, self.female_fx, self.small_males_fx)
            )

            index_fx = np.argsort(self.fx[0])

            self.fx, self.pop = self.fx[:, index_fx], self.pop[index_fx, :]

            best_indiv = self.pop[0, :]
            opt_val = self.fx[0, 0]

            fopt = np.hstack((fopt, opt_val))
            fmean = np.hstack((fmean, np.mean(self.fx)))
            evo_pop_size = np.hstack((evo_pop_size, self.pop_size))

            if opt_val < self.one_elit_fx:
                gen_improve += 1
                improve_rate = gen_improve / gen
                self.one_elit_fx = opt_val

            if opt_val <= self.fthreshold_fx:
                is_global = 1
                break

            if gen == max_gen_exam1:
                if improve_rate < 0.5:
                    is_global = 0
                    break

        # best_indiv = np.zeros(self.dimension)
        # opt_val = 0.0
        # num_eva = self.max_num_eva
        # fopt = [0.0] * num_eva
        # fmean = [0.0] * num_eva
        proc_time = 0.0
        # evo_pop_size = [self.pop_size] * num_eva
        return best_indiv, opt_val, num_eva, fopt, fmean, proc_time, evo_pop_size
