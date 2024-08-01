import numpy as np


class BenchmarkFunctions:
    @staticmethod
    def get_function(function_id: int) -> tuple:
        if function_id == 1:
            return BenchmarkFunctions.function1()
        # Add more functions as needed
        else:
            raise ValueError(f"Function ID {function_id} not implemented")

    @staticmethod
    def function1() -> tuple:
        nvar = 50
        ra = 100.0
        rb = -100.0
        fthreshold_fx = 0.0
        return nvar, ra, rb, fthreshold_fx

    @staticmethod
    def evaluate(x: np.ndarray, function_id: int) -> float:
        if function_id == 1:
            return np.sum(x**2)  # Example: sphere function
        # Add more evaluation functions as needed
        else:
            raise ValueError(f"Function ID {function_id} not implemented")
