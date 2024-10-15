import numpy as np
from scipy.special import gamma
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

        for nn in range(1, ps + 1, 4):
            if ps - nn >= 4:
                n_loc = 4
            else:
                n_loc = ps - nn + 1

            ss = 0

            while ss <= n_loc - 1:
                temp = np.zeros((1, self.nvar))
                for i in range(0, math.floor(self.nvar / 2)):
                    temp[:, i] = self.rb[0, i] + (self.ra[0, i] - self.rb[0, i]) * (
                        f1[ss] + ((np.random.rand() * 2) - 1) * 0.01
                    )

                for i in range(math.floor(self.nvar / 2), self.nvar):
                    temp[:, i] = self.rb[0, i] + (self.ra[0, i] - self.rb[0, i]) * (
                        f2[ss] + ((np.random.rand() * 2) - 1) * 0.01
                    )

                x[individu_X, :] = temp
                individu_X += 1
                ss += 1

        return x

    def move_big_males_female_first_stage(self):
        """
        Move Big Males and Female in the first stage. The winner mates Female (if the Female wants)

        Raises:
            ValueError: array shape is not correct
        """
        hq = np.copy(self.big_males)
        hq_fx = np.copy(self.big_males_fx)

        temp_small_males = np.copy(self.big_males)
        temp_small_males_fx = np.copy(self.big_males_fx)

        for ss in range(temp_small_males.shape[0]):
            max_fol_hq = np.random.randint(1, 3)  # generate random [1,2]
            vm = np.zeros((1, self.nvar))
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

            # new movement of big males
            new_big_males = np.copy(temp_small_males[ss, :]) + np.copy(vm)
            new_big_males = self.trimr(new_big_males)

            temp_small_males[ss, :] = np.copy(new_big_males)
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
            fx1 = self.evaluation(offsprings[0, :].reshape(1, -1)).reshape(1, -1)
            fx2 = self.evaluation(offsprings[1, :].reshape(1, -1)).reshape(1, -1)

            # keep the best position of female
            if fx1 < fx2:
                if fx1 < self.female_fx:
                    self.female = offsprings[0, :].copy()
                    self.female_fx = fx1.copy()
            else:
                if fx2 < self.female_fx:
                    self.female = offsprings[1, :].copy()
                    self.female_fx = fx2.copy()
        else:
            # asexual reproduction
            new_female = self.mutation()

            if new_female.shape != (1, self.nvar):
                new_female = new_female.reshape(1, -1)

                if new_female.shape() != (1, self.nvar):
                    raise ValueError("Array female after mutation is not correct")

            fx = self.evaluation(new_female)

            if fx < self.female_fx:
                self.female = new_female.copy()
                self.female_fx = fx.copy()

    def move_big_males_female_second_stage(self):
        """
        Move Big Males and Female in the second stage. The winner mates Female (if the Female wants)

        Raises:
            ValueError: array shape is not correct

        """

        if self.all_hq.shape[0] != 0:
            global_hq = np.vstack((np.copy(self.big_males), np.copy(self.all_hq)))
            global_hq_fx = np.hstack(
                (np.copy(self.big_males_fx), np.copy(self.all_hq_fx))
            )
        else:
            global_hq = np.copy(self.big_males)
            global_hq_fx = np.copy(self.big_males_fx)

        temp_small_males = np.copy(self.big_males)
        temp_small_males_fx = np.copy(self.big_males_fx)

        for ss in range(temp_small_males.shape[0]):
            max_fol_hq = np.random.randint(1, 3)
            vm = np.zeros((1, self.nvar))
            rhq = np.random.permutation(global_hq.shape[0])
            fol_hq = 0

            for fs in range(len(rhq)):
                ind = rhq[fs]
                if ind != ss:
                    # Semi randomly select sn individual to define an attraction or a distraction
                    if (global_hq_fx[:, ind] < temp_small_males_fx[:, ss]) or (
                        np.random.rand() < 0.5
                    ):
                        vm += np.random.rand() * (
                            global_hq[ind, :] - temp_small_males[ss, :]
                        )
                    else:
                        vm += np.random.rand() * (
                            temp_small_males[ss, :] - global_hq[ind, :]
                        )

                    fol_hq += 1
                    if fol_hq >= max_fol_hq:
                        break

            new_big_males = temp_small_males[ss, :].copy() + vm.copy()
            new_big_males = self.trimr(new_big_males.copy())

            temp_small_males[ss, :] = new_big_males.copy()
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
            fx1 = self.evaluation(offsprings[0, :].reshape(1, -1)).reshape(1, -1)
            fx2 = self.evaluation(offsprings[1, :].reshape(1, -1)).reshape(1, -1)

            # keep the best position of female
            if fx1 < fx2:
                if fx1 < self.female_fx:
                    self.female = offsprings[0, :].copy()
                    self.female_fx = fx1.copy()
            else:
                if fx2 < self.female_fx:
                    self.female = offsprings[1, :].copy()
                    self.female_fx = fx2.copy()
        else:
            # asexual reproduction
            new_female = self.mutation()

            if new_female.shape != (1, self.nvar):
                new_female = new_female.reshape(1, -1)

                if new_female.shape() != (1, self.nvar):
                    raise ValueError("Array female after mutation is not correct")

            fx = self.evaluation(new_female)

            if fx < self.female_fx:
                self.female = new_female.copy()
                self.female_fx = fx.copy()

        if self.female.shape != (1, self.nvar):
            self.female = self.female.reshape(1, -1)
            if self.female.shape != (1, self.nvar):
                raise ValueError("Female second stage shape is not (1,self.nvar)")

        if self.female_fx.shape != (1, 1):
            self.female_fx = self.female_fx.reshape(1, -1)
            if self.female_fx.shape != (1, 1):
                raise ValueError("Female second stage shape is not (1,1")

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
        """
        Move (Mlipir) Small Males aside with MlipirRate = 0.5 to do a low-exploitative high-explorative searching


        Params:


        Returns:


        Raises:
            ValueError: array shape is not correct
        """

        if self.all_hq.shape[0] != 0:
            hq = np.copy(np.vstack((self.big_males, self.all_hq)))
        else:
            hq = np.copy(self.big_males)

        temp_weak_males = np.copy(self.small_males)
        temp_weak_males_fx = np.copy(self.small_males_fx)

        for ww in range(self.small_males.shape[0]):
            max_fol_hq = np.random.randint(1, 3)
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

        max_step = self.mut_radius * (
            self.ra - self.rb
        )  # Maximum step of the Female mutation

        for i in range(self.nvar):
            if (
                np.random.rand() < self.mut_rate
            ):  #  Check if a random value is lower than the Mutation Rate
                new_female[:, i] = (
                    self.female[:, i] + ((2 * np.random.rand()) - 1) * max_step[0, i]
                )

        new_female = self.trimr(
            new_female
        )  # Limit the values into the given dimensional boundaries
        return new_female

    def levy(self, beta=1.0, size=None) -> np.ndarray:
        """
        Generate Lévy flight steps.

        Parameters:
        beta (float): Power law index (should be between 1 and 2)
        size (tuple, list): size of levy flights steps, for instance: (3,2), 5, (4, )

        Returns:
        np.ndarray: Array of shape (n, m) contains Lévy flight steps
        """
        # Calculate parameters
        num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)  # numerator
        den = (
            math.gamma((1 + beta) / 2) * beta * np.power(2, ((beta - 1) / 2))
        )  # denominator
        sigma_u = (num / den) ** (1 / beta)  # standard deviation
        u = np.random.normal(0, sigma_u, size)

        # sigma_v : standard deviation of v
        sigma_v = 1
        v = np.random.normal(0, sigma_v, size)

        # Calculate Lévy steps
        z = u / np.power(np.abs(v), (1 / beta))

        return z

    def adding_pop(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Adding an individual randomly

        Params:
            x : current individuals (np.ndarray) with shape of (1, self.nvar)

        Returns:
            Returns new individual that being added to new population

        Raise:
            ValueError: levy step, new_x, or new_fx array shape is not correct

        """
        # generate levy flight step
        new_x_temp = x.copy()

        levy_value = 0.05 * self.levy(1.5, (1, self.nvar))

        new_x = new_x_temp + (levy_value * np.abs(self.ra - self.rb))

        if new_x.shape != (1, self.nvar):
            new_x = new_x.reshape(1, -1)
            if new_x.shape != (1, self.nvar):
                raise ValueError(
                    f"new_x shape with a shape of {new_x.shape} in adding_pop is not correct. Must be (1,self.nvar)"
                )

        new_x = self.trimr(new_x)

        new_fx = self.evaluation(new_x)

        if new_fx.shape != (1, 1):
            new_fx = new_fx.reshape(1, -1)
            if new_fx.shape != (1, 1):
                raise ValueError(
                    f"new_fx with a shape of {new_fx.shape} shape in adding_pop is not correct. Must be (1,1)"
                )

        return new_x, new_fx

    def reposition(self, x: np.ndarray, fx: float) -> Tuple[np.ndarray, float]:
        """
        Reposition an individual randomly

        Params:
            x: komodo individuals (np.ndarray) with shape (1, self.nvar)
            fx: fitness value of komodo x (float) with shape (1,1)

        Returns:
            Returns a new position of the individual x (new_x) and its fitness value (new_fx)

        Raises:
            ValueError: temp_x, or temp_fx array shape is not correct


        """
        temp_x = np.ones((1, self.nvar)) * x.copy()
        max_step = self.mut_radius * (self.ra - self.rb)

        for ii in range(self.nvar):
            if np.random.rand() < self.mut_rate:
                temp_x[:, ii] = (
                    x[:, ii]
                    + ((2 * np.random.rand()) - 1) * self.mut_radius * max_step[0, ii]
                )

        if temp_x.shape != (1, self.nvar):
            temp_x = temp_x.reshape(1, -1)
            if temp_x.shape != (1, self.nvar):
                raise ValueError(
                    f"temp_x shape with a shape of {temp_x.shape} in reposition is not correct. Must be (1,self.nvar)"
                )

        temp_x = self.trimr(temp_x)

        temp_fx = self.evaluation(temp_x)

        if temp_fx.shape != (1, 1):
            temp_fx = temp_fx.reshape(1, -1)
            if temp_fx.shape != (1, 1):
                raise ValueError(
                    f"temp_fx with a shape of {temp_fx.shape} in reposition is not correct. Must be (1,1)"
                )

        if temp_fx < fx:  # TempFX is better than the original FX
            new_x = temp_x
            new_fx = temp_fx
        else:  # TempFX is worse than or equal to the original FX
            new_x = x
            new_fx = fx

        return new_x, new_fx

    def replacement(
        self, x: np.ndarray, fx: np.ndarray, y: np.ndarray, fy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Replacement: sort the old and new populations and select the best ones

        Parameters:
        x  : old population of LX individuals with the shape of (1,nvar)
        fx : old fitness with the shape of (1,nvar)
        y  : new population with the shape of (1,nvar)
        fy : new fitness with the shape of (1,nvar)

        Returns:
        z  : survivor of LX individuals with the shape of (1,nvar)
        fz : survivor fitness with the shape of (1,nvar)

        Raises:

        """

        lx = x.shape[0]  # Number of individuals in old population
        xy = np.vstack((x, y))  # Joint individuals of old and new population
        fxfy = np.hstack((fx, fy))  # Joint fitness values of old and new population

        # Sort all fitness values and get sorted indices
        sorted_ind = np.argsort(fxfy[0])
        sorted_value = np.sort(fxfy)

        # Select the best individuals
        z = np.copy(xy[sorted_ind[:lx], :])
        fz = np.copy(sorted_value[:, :lx])

        return z, fz

    def trimr(self, x: np.ndarray) -> np.ndarray:
        """
        Limit the values into the given dimensional boundaries

        Params:
            x: array of shape (n,nvar) of Komodo individuals

        Returns:
            Returns new array of komodo individuals with shape (1,nvar) but have trimmed to dimensional boundaries

        Raises:

        """

        # # x -> clipped
        # # transfer
        # def wrapper(x):
        #     # wrap a real value to its binary
        #     # 1. pure KMA
        #     # 2. Regular binary wrapper -> v-shaped and s-shaped
        #     # 3. Time variant S-shaped

        #     return 0

        return np.clip(x, self.rb, self.ra)

    def run(
        self,
    ) -> Tuple[np.ndarray, float, int, List[float], List[float], float, List[int]]:

        # generate the initial population of population size komodo individuals
        self.pop = self.pop_cons_initialization(self.pop_size)

        # calculate all the fitness values of the population size komodo individuals
        self.fx = np.zeros((1, self.pop_size))

        for i in range(self.pop_size):
            self.fx[:, i] = self.evaluation(self.pop[i, :])

        # print(self.fx)

        # sort the fitness values of the initial population
        index_fx = np.argsort(self.fx[0])

        self.fx, self.pop = self.fx[:, index_fx], self.pop[index_fx, :]

        self.one_elit_fx = self.fx[0, 0]  # the best so-far

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
                raise ValueError(
                    f"Female with a shape of {self.female.shape} in main function first stage is incorrect. Must be (1,nvar)"
                )

            self.small_males = np.copy(self.pop[self.num_BM + 1 :, :])
            self.small_males_fx = np.copy(self.fx[:, self.num_BM + 1 :])

            # move big makes and female
            self.move_big_males_female_first_stage()
            # move small males
            self.move_small_males_first_stage()

            self.pop = np.vstack((self.big_males, self.female, self.small_males)).copy()

            # make sure female_fx has the same dimension
            if self.female_fx.shape != (1, 1):
                self.female_fx = self.female_fx.reshape(1, -1)
                if self.female_fx.shape != (1, 1):
                    raise ValueError(
                        f"Female_fx with a shape of {self.female_fx.shape} in main function first stage is incorrect. Must be (1,1)"
                    )

            self.fx = np.hstack(
                (self.big_males_fx, self.female_fx, self.small_males_fx)
            )

            index_fx = np.argsort(self.fx[0])

            self.fx, self.pop = (
                self.fx[:, index_fx].copy(),
                self.pop[index_fx, :].copy(),
            )

            best_indiv = self.pop[0, :].copy()
            opt_val = np.min(self.fx[0]).copy()

            fopt = np.hstack((fopt, opt_val))
            fmean = np.hstack((fmean, np.mean(self.fx[0])))
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

        #
        #
        # Second Stage
        #
        #

        if (not is_global) and num_eva <= self.max_num_eva:
            first_stage_pop = np.copy(self.pop)
            first_stage_pop_fx = np.copy(self.fx)
            swarm_size = first_stage_pop.shape[0]
            self.num_BM = int(np.floor(swarm_size / 2))

            increment_ada_pop_size = swarm_size
            decrement_ada_pop_size = swarm_size

            self.mlipir_rate = 0.5
            max_gen_improve = 2
            max_gen_stagnan = 2

            gen_improve = 0
            gen_stagnan = 0

            cons_pop = self.pop_cons_initialization(self.max_ada_pop_size - swarm_size)

            cons_pop_fx = np.zeros((1, cons_pop.shape[0]))

            for i in range(cons_pop.shape[0]):
                individu = cons_pop[i, :]
                result = self.evaluation(individu)
                cons_pop_fx[:, i] = result

            self.pop = np.vstack((first_stage_pop, cons_pop)).copy()
            self.pop_size = self.pop.shape[0]
            self.fx = np.hstack((first_stage_pop_fx, cons_pop_fx)).copy()
            self.one_elit_fx = np.min(self.fx[0])

            while num_eva < self.max_num_eva:
                ada_pop_size = self.pop.shape[0]
                self.all_hq = np.zeros((ada_pop_size, self.nvar))
                self.all_hq_fx = np.zeros((1, ada_pop_size))

                for ind in range(0, ada_pop_size, swarm_size):
                    micro_swarm = np.copy(self.pop[ind : ind + swarm_size, :])
                    micro_swarm_fx = np.copy(self.fx[:, ind : ind + swarm_size])

                    index_fx = np.argsort(micro_swarm_fx[0])
                    micro_swarm = np.copy(micro_swarm[index_fx, :])

                    micro_swarm_fx = np.copy(micro_swarm_fx[:, index_fx])

                    self.all_hq = np.copy(
                        np.vstack((self.all_hq, micro_swarm[: self.num_BM, :]))
                    )
                    self.all_hq_fx = np.copy(
                        np.hstack((self.all_hq_fx, micro_swarm_fx[:, : self.num_BM]))
                    )

                for ind in range(0, ada_pop_size, swarm_size):
                    micro_swarm = np.copy(self.pop[ind : ind + swarm_size, :])
                    micro_swarm_fx = np.copy(self.fx[:, ind : ind + swarm_size])

                    index_fx = np.argsort(micro_swarm_fx[0])
                    micro_swarm = np.copy(micro_swarm[index_fx, :])
                    micro_swarm_fx = np.copy(micro_swarm_fx[:, index_fx])

                    self.big_males = np.copy(micro_swarm[: self.num_BM, :])
                    self.big_males_fx = np.copy(micro_swarm_fx[:, : self.num_BM])

                    self.female = np.copy(micro_swarm[self.num_BM, :]).reshape(1, -1)
                    self.female_fx = np.copy(micro_swarm_fx[:, self.num_BM]).reshape(
                        1, -1
                    )

                    if self.female.shape != (1, self.nvar):
                        raise ValueError(
                            f"Female shape with a shape of {self.female.shape} is incorrect. Must be (1,{self.nvar})"
                        )

                    self.small_males = np.copy(micro_swarm[self.num_BM + 1 :, :])
                    self.small_males_fx = np.copy(micro_swarm_fx[:, self.num_BM + 1 :])

                    self.move_big_males_female_second_stage()
                    self.move_small_males_second_stage()

                    self.all_hq[ind : ind + self.num_BM, :] = np.copy(self.big_males)
                    self.all_hq_fx[:, ind : ind + self.num_BM] = np.copy(
                        self.big_males_fx
                    )

                    # resulted new population
                    self.pop[ind : ind + swarm_size, :] = np.vstack(
                        (
                            np.copy(self.big_males),
                            np.copy(self.female),
                            np.copy(self.small_males),
                        )
                    )

                    self.fx[:, ind : ind + swarm_size] = np.hstack(
                        (
                            np.copy(self.big_males_fx),
                            np.copy(self.female_fx),
                            np.copy(self.small_males_fx),
                        )
                    )

                    num_eva += swarm_size

                    opt_val = np.min(self.fx[0])

                    if opt_val <= self.fthreshold_fx:
                        break

                # random population
                ind = np.random.permutation(len(self.fx[0]))
                self.pop = np.copy(self.pop[ind, :])
                self.fx = np.copy(self.fx[:, ind])

                best_indiv = self.pop[np.argmin(self.fx[0]), :]

                opt_val = np.min(self.fx[0])
                fopt = np.hstack((fopt, np.copy(opt_val)))
                fmean = np.hstack((fmean, np.mean(self.fx[0])))

                # if roundup(opt_val) <= ftresholdfx
                if opt_val <= self.fthreshold_fx:
                    break

                ######################################
                # Self-adaptation of population size #
                ######################################

                if opt_val < self.one_elit_fx:
                    gen_improve += 1
                    gen_stagnan = 0
                    self.one_elit_fx = opt_val
                else:
                    gen_stagnan += 1
                    gen_improve = 0

                # If consecutive fitness values show an improvement
                if gen_improve > max_gen_improve:
                    ada_pop_size -= decrement_ada_pop_size
                    if ada_pop_size < self.min_ada_pop_size:
                        ada_pop_size = self.min_ada_pop_size

                    sorted_ind = np.argsort(self.fx[0])
                    sorted_pop = self.pop[sorted_ind, :]
                    self.pop = sorted_pop[:ada_pop_size, :]
                    self.fx = self.fx[:, sorted_ind]
                    self.fx = self.fx[:, :ada_pop_size]
                    gen_improve = 0

                # if consecutive fitness values show a stagnation
                if gen_stagnan > max_gen_stagnan:
                    ada_pop_size_old = self.pop.shape[0]
                    ada_pop_size += increment_ada_pop_size
                    num_add_pop = ada_pop_size - ada_pop_size_old

                    if ada_pop_size > self.max_ada_pop_size:
                        ada_pop_size = ada_pop_size_old
                        num_add_pop = 0  # No individual added into the population

                    if ada_pop_size > ada_pop_size_old:
                        new_pop = np.zeros((num_add_pop, self.nvar))
                        new_pop_fx = np.zeros((1, num_add_pop))

                        for nn in range(num_add_pop):
                            new_pop[nn, :], new_pop_fx[:, nn] = self.adding_pop(
                                best_indiv
                            )

                        self.pop = np.vstack((self.pop, new_pop)).copy()
                        self.fx = np.hstack((self.fx, new_pop_fx)).copy()

                        num_eva += num_add_pop
                    else:  # no adding population
                        for nn in range(self.pop.shape[0]):
                            self.pop[nn, :], self.fx[:, nn] = self.reposition(
                                np.copy(self.pop[nn, :]).reshape(1, -1),
                                np.copy(self.fx[:, nn]).reshape(1, -1),
                            )
                        num_eva += self.pop.shape[0]
                    gen_stagnan = 0

                rand_ind = np.random.permutation(self.pop.shape[0])
                self.fx = self.fx[:, rand_ind]
                self.pop = self.pop[rand_ind, :]

                evo_pop_size = np.hstack((evo_pop_size, ada_pop_size))
                gen += 1
        # best_indiv = np.zeros(self.dimension)
        # opt_val = 0.0
        # num_eva = self.max_num_eva
        # fopt = [0.0] * num_eva
        # fmean = [0.0] * num_eva
        proc_time = 0.0
        # evo_pop_size = [self.pop_size] * num_eva
        return best_indiv, opt_val, num_eva, fopt, fmean, proc_time, evo_pop_size
