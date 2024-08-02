import numpy as np


class Benchmark:
    @staticmethod
    def get_function(function_id: int) -> tuple:
        return Benchmark.get_params(function_id)

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
                ub = 100.0
                lb = -100.0
                fthreshold_fx = 0.0

            case _:
                # default
                raise ValueError(f"Function ID {function_id} not implemented")

        return n_var, ub, lb, fthreshold_fx

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

        if x.ndim != 2 and x.shape[0] == 1:
            raise Exception("Shape of the array X is not (1,n_var)")

        n_dimension = x.shape[0]

        match (function_id):
            case 1:
                # Sphere function
                return np.sum(x**2)
            case _:
                raise ValueError(f"Function ID {function_id} not implemented")
