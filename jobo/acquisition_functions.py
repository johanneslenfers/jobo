from typing import List

import numpy as np
import random

from jobo.util import AcquisitionFunction
from jobo.util import SurrogateModel


class RandomSampling(AcquisitionFunction):
    def __init__(self, budget: int):
        self.budget = budget

    def optimize(self, surrogate_model: SurrogateModel):

        # random sample surrogate model and return minimum 
        minimum: float = random.uniform(surrogate_model.range.min, surrogate_model.range.max)
        for _ in range(self.budget):
            candidate: float = random.uniform(surrogate_model.range.min, surrogate_model.range.max) 
            if (surrogate_model.evaluate(candidate) < surrogate_model.evaluate(minimum)):
                minimum = candidate

        return minimum


class GridSearch(AcquisitionFunction):
    def __init__(self, budget: int):
        self.budget = budget

    # TODO handle sampling of same points 
    def optimize(self, surrogate_model: SurrogateModel):

        x = np.linspace(
            start = surrogate_model.range.min,
            stop = surrogate_model.range.max,
            num = self.budget
            )

        minimum_index = 0
        for i in x:
            if(surrogate_model.evaluate(i) < surrogate_model.evaluate(minimum_index)):

                minimum_index = i

        return minimum_index
