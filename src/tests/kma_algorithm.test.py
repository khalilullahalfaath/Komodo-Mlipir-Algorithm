import unittest
import numpy as np
from io import StringIO
import sys
import os
from os.path import dirname, join as pjoin
import scipy.io as sio


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.algorithms.kma_algorithm import KMA


class TestKMA(unittest.TestCase):
    def setUp(self):
        self.function_id = 1
        self.dimension = 50
        self.max_num_eva = 1000
        self.pop_size = 5

        self.kma = KMA(
            self.function_id, self.dimension, self.max_num_eva, self.pop_size
        )

    def test_fx_size(self):
        # Run the initialization and evaluation
        self.kma.pop = self.kma.pop_cons_initialization(self.kma.pop_size)
        self.kma.fx = np.zeros((1, self.kma.pop_size))
        for i in range(self.kma.pop_size):
            self.kma.fx[:, [i]] = self.kma.evaluation(self.kma.pop[[i], :])

        # Check the size of fx
        self.assertEqual(self.kma.fx.shape, (1, 5), "fx should have shape (1, 5)")

    def test_population_and_fx_values(self):
        np.random.seed(42)  # Set a fixed seed for reproducibility

        # Generate the population
        self.kma.pop = self.kma.pop_cons_initialization(self.kma.pop_size)

        print("Generated population:")
        print(self.kma.pop)

        mat_file_path = os.path.join(os.path.dirname(__file__), "data", "pop.mat")
        mat_contents = sio.loadmat(mat_file_path)

        print(mat_contents)

        # Convert the MATLAB data string to a numpy array
        matlab_pop = mat_contents

        print("\nExpected population from MATLAB:")
        print(matlab_pop)

        print("\nDifferences in population:")
        pop_diff = self.kma.pop - matlab_pop
        print(pop_diff)

        # Check if the populations are close
        np.testing.assert_allclose(
            self.kma.pop,
            matlab_pop,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Generated population does not match expected MATLAB output",
        )

        print("\nPopulation check passed. Now evaluating fitness...")

        # Evaluate fitness
        self.kma.fx = np.zeros((1, self.kma.pop_size))
        for i in range(self.kma.pop_size):
            self.kma.fx[:, [i]] = self.kma.evaluation(self.kma.pop[[i], :])
            print(f"\nEvaluation {i+1}:")
            print(f"Individual: {self.kma.pop[i, :]}")
            print(f"Fitness: {self.kma.fx[0, i]}")

        # Paste your MATLAB fitness data here
        matlab_fx = np.array(
            [
                4.788615799813203e05,
                4.795563179041033e05,
                4.819387031017045e05,
                4.795778390078847e05,
                4.816324607430795e05,
            ]
        )

        print("\nPython fx:", self.kma.fx[0])
        print("MATLAB fx:", matlab_fx)

        differences = self.kma.fx[0] - matlab_fx
        print("\nAbsolute differences in fitness:", differences)
        print("Relative differences in fitness:", differences / matlab_fx)

        np.testing.assert_allclose(
            self.kma.fx[0],
            matlab_fx,
            rtol=1e-3,
            atol=1,
            err_msg="fx values do not match expected MATLAB output",
        )


if __name__ == "__main__":
    unittest.main()
