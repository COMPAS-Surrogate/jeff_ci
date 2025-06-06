import numpy as np

from ..lnl_computer import LnLComputer  
from .active_learner import ActiveLearner  

class LnLSurrogate:
    def __init__(self):
        self.lnl_computer = LnLComputer()
        self.reference_lnl = self._compute_reference_lnl()
        self.active_learner = ActiveLearner(self.negative_lnl)

    def _compute_reference_lnl(self, num_samples=10):
        """
        Compute the reference log-likelihood by sampling multiple times and taking the mean.
        """
        samples = [self.lnl_computer.sample() for _ in range(num_samples)]
        return np.mean(samples)

    def negative_lnl(self, x):
        """
        Compute the negative log-likelihood relative to the reference point.
        """
        lnl = self.lnl_computer.compute(x)
        return -(lnl - self.reference_lnl)

    def predict(self, x):
        """
        Predict using the ActiveLearner and adjust the result relative to the reference log-likelihood.
        """
        gp_y = self.active_learner.predict(x)
        return self.reference_lnl - gp_y