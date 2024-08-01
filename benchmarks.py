import numpy as np


class BenchmarkFunctions:
    @staticmethod
    def get_function(function_id: int) -> tuple:
        return BenchmarkFunctions.get_params(function_id)

    def get_params(function_id: int) -> tuple[int, float, float, float]:
        """
        Get params of a function

        Args:
            function_id: the id of a function

        Returns:
            Returns the number of variables (int), upper bound value (float),  lower_bound value (float), and function treshold of fx (float)

        Raises:
            ValueError: function_id not found
        """
        match (function_id):
            case 1:
                n_var = 50
                ra = 100.0
                rb = -100.0
                fthreshold_fx = 0.0

            case _:
                # default
                raise ValueError(f"Function ID {function_id} not implemented")

        return n_var, ra, rb, fthreshold_fx

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

        if x.ndim != 1:
            raise Exception("Shape off the array X is not 1 dimension")

        n_dimension = x.shape[0]

        match (function_id):
            case 1:
                # Sphere function
                return np.sum(x**2)
            case _:
                raise ValueError(f"Function ID {function_id} not implemented")
