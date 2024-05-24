from typing import Callable
from typing import List

import warnings
import numpy as np

from jobo.util import SurrogateModel
from jobo.util import Sample
from jobo.util import Range


class PolyomialModel(SurrogateModel):

    def __init__(self, range: Range, degree: int):
        self.range = range
        self.degree: int = degree
        self.surrogate: Callable = id

        # disable RankWarnings globally 
        warnings.filterwarnings('ignore', category=np.RankWarning)


    def update(self, samples: List[Sample]):

        # least squares polynomial fit/regression 
        coeffs = np.polyfit(
            x = [elem.x for elem in samples],
            y = [elem.y for elem in samples],
            deg=self.degree
            )
        self.surrogate = np.poly1d(coeffs)

    def evaluate(self, x: float) -> float:
        return  self.surrogate(x)

    def id(self, x: float) -> float:
        return x