from typing import Callable

import numpy as np

from jobo.util import Range
from jobo.util import ObjectiveFunction

from jobo.surrogate_models import PolyomialModel
from jobo.acquisition_functions import RandomSampling

from jobo.jobo import JoBO


# example objective functions 
def function_polynom(x: float) -> float:
    return -x**4+3*x**3+100*x**2-20*x+80

range_polynom: Range = Range(
    min = -10,
    max =  10,
    )

def function_swing(x: float) -> float:
    y: float = 0
    term1: float = x*np.cos(np.sin(abs(x**2 - y**2)))**2 - 0.5
    term2: float = (1 + 0.001*(x**2 + y**2))**2
    f: float = 0.5 + term1 / term2
    return f

range_swing: Range = Range(
    min = - 5,
    max = 5
    )


def main():

    # example_function = function_polynom
    # example_range = range_polynom

    example_function: Callable = function_swing
    example_range: Range = range_swing

    jobo: JoBO = JoBO(
            initial_points=1,
            iterations=25,
            objective_function=ObjectiveFunction(
                function=example_function, 
                range = example_range
            ),
            surrogate_model=PolyomialModel(
                range = example_range,
                degree = 10, 
            ),
            acquisition_function=RandomSampling(
                budget=100
            ),
            plotting_precision = 1000
        )

    jobo.run()


if __name__ == "__main__":
    main()

