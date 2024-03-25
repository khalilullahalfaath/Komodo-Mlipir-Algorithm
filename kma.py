from get_function import get_function

# libraries
import numpy as np
import math


class KMA:
    def __init__(
        self, function_id=1, dimension=50, max_eval_number=25000, population_size=5
    ) -> None:
        self.function_id = function_id
        self.dimension = dimension
        self.max_eval_number = max_eval_number
        self.pop_size = population_size  # population size

        # for the second stage
        self.min_adaptive_pop_size = self.pop_size * 4
        self.max_adaptive_pop_size = self.pop_size * 40

        # for evaluation
        self.upper_bound = None
        self.lower_bound = None
        self.optimum_value = None
        self.var_num = None

    def set_function(self):
        """
        set a function to be evaluated by KMA
        """
        self.upper_bound, self.lower_bound, self.optimum_value, self.var_num = (
            get_function(self.function_id, self.dimension)
        )

        self.upper_bound = np.ones((1, self.dimension)) * self.upper_bound
        self.lower_bound = np.ones((1, self.dimension)) * self.lower_bound

    def generate_population(self):
        """
        Generate population based on upper and lower bounds
        """
        f1 = np.array([0.01, 0.01, 0.99, 0.99])
        f2 = np.array([0.01, 0.99, 0.01, 0.99])
        population = np.zeros((self.pop_size, self.var_num))
        pop_indx = 0

        for nn in range(1, self.pop_size + 1, 4):
            if self.pop_size - nn >= 4:
                NL = 4
            else:
                NL = self.pop_size - nn + 1

            ss = 0
            while ss <= NL - 1:
                Temp = np.zeros((1, self.dimension))
                for i in range(0, math.floor(self.dimension / 2)):
                    Temp[:, i] = self.lower_bound[0, i] + (
                        (self.upper_bound[0, i] - self.lower_bound[0, i])
                        * (f1[ss] + ((np.random() * 2) - 1) * 0.01)
                    )

                for i in range(math.floor(self.dimension / 2), self.dimension):
                    Temp[:, i] = self.lower_bound[0, i] + (
                        (self.upper_bound[0, i] - self.lower_bound[0, i])
                        * (f2[ss] + ((np.random() * 2) - 1) * 0.01)
                    )

                population[pop_indx, :] = Temp
                pop_indx += 1
                ss += 1
        return population


my_KMA = KMA()
