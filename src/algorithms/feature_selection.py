# 0. input: binary value
#
# output: hasil evaluasi feature -> dataset pake random forest

import numpy as np


class Benchmark:
    @staticmethod
    def get_function(dimension: int, function_id: int) -> tuple:
        return Benchmark.get_params(dimension, function_id)

    def get_params(dimension: int, function_id: int) -> tuple[int, float, float, float]:
        """
        Get params of a function

        Args:
            dimension: the number of dimension default equals to dimension
            function_id: the id of a function

        Returns:
            Returns the number of variables (int), upper bound value (float),  lower_bound value (float), and function treshold of fx (float)

        Raises:
            ValueError: function_id not found
        """
        n_var = dimension

    @staticmethod
    def evaluate(x: np.ndarray, function_id: int) -> float:
        """
        Evaluate the komodo individual on the given function

        Args:
            x: Komodo individual consists of numpy array with a shape of ((n_var))
            function_id: the id of a function to be evaluated

        Returns:
            Returns function_id evaluation result (float)

        Raises:
            ValueError: function_id not found
            IndexOutOfRange
            DivideByZero
        """

        if x.ndim != 2 and x.shape[0] != 1:
            x = x.reshape(1, -1)
            if x.ndim != 2 and x.shape[0] != 1:
                raise Exception("Shape of the array X is not (1,n_var)")

        dim = x.shape[1]

        # feature selection
