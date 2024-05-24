from typing import Callable
from typing import List

from abc import ABC, abstractmethod


class Range:
    def __init__(self, min: int, max: int):
        self.min = min
        self.max = max


class Sample:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    

class Function(ABC):

    def __init__(self, range: Range):
        self.range = range

    @abstractmethod
    def evaluate(self, x: float) -> float:
        pass


class ObjectiveFunction(Function):

    def __init__(self, range: Range, function: Callable):
        self.range = range
        self.function = function

    def evaluate(self, x: float) -> float:
        return self.function(x)


class SurrogateModel(Function):

    @abstractmethod
    def update(self, samples: List[Sample]) -> float:
        pass


class AcquisitionFunction(ABC):

    @abstractmethod
    def optimize(self, surrogate_model: Function) -> float:
        pass


class BoState:

    def __init__(self, objective_function: ObjectiveFunction, surrogate_model: SurrogateModel, samples: List[Sample]):
        self.objective_function = objective_function
        self.surrogate_model = surrogate_model
        self.samples = samples
