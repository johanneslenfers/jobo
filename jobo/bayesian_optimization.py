from typing import Optional
from typing import Callable
from typing import List

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import random
import copy

from jobo.util import Range
from jobo.util import ObjectiveFunction
from jobo.util import SurrogateModel
from jobo.util import Sample
from jobo.util import BoState

from jobo.acquisition_functions import AcquisitionFunction


class BayesianOptimization:

    def __init__(self, 
                initial_points: int, 
                iterations: int, 
                objective_function: ObjectiveFunction, 
                surrogate_model: SurrogateModel, 
                acquisition_function: AcquisitionFunction,
                y_plot_limit: Optional[dict[str, int]] = None):

        self.initial_points = initial_points
        self.iterations = iterations
        self.objective_function = objective_function
        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.y_plot_limit = y_plot_limit
        self.samples = [[], []]

        # plotting specifics 
        self.model_plot = None
        self.objective_plot = None
        self.eval_points_plot = None
        self.fig = None

        # save state of each optimization step 
        self.record: List[BoState] = []


    def initialize(self):

        # random sample initial points 
        design_of_experiment: List[float] = [(random.uniform(self.objective_function.range.min, self.objective_function.range.max)) for _ in range(self.initial_points)]

        for sample in design_of_experiment: 

            # evaluate initial points and update model 
            value: float = self.objective_function.evaluate(sample)

            self.samples[0].append(sample)
            self.samples[1].append(value)

            samples = [Sample(x,y) for x, y in zip(self.samples[0], self.samples[1])]

            self.surrogate_model.update(samples)

            # add state to record 
            self.record.append(
                BoState(objective_function=self.objective_function, 
                        surrogate_model=copy.copy(self.surrogate_model), 
                        samples=samples)
                )

    def optimize(self):

        # bayesian optimization main loop 
        for _ in range(self.iterations):

            # determine next point 
            next_point = self.acquisition_function.optimize(
                surrogate_model=self.surrogate_model)

            # evaluate next point and update model 
            value: float = self.objective_function.evaluate(next_point)

            self.samples[0].append(next_point)
            self.samples[1].append(value)

            samples = [Sample(x,y) for x, y in zip(self.samples[0], self.samples[1])]

            self.surrogate_model.update(samples)

            # add state to record 
            self.record.append(
                BoState(objective_function=self.objective_function, 
                        surrogate_model=copy.copy(self.surrogate_model), 
                        samples=samples)
                )
