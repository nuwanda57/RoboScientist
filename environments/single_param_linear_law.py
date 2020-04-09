import environments.base as base
import lib.success_criterion as sc
import torch


class LinearLawEnvironment(base.EnvironmentBase):
    """
    Environment a*x + b.
    """
    def __init__(self, a: float, b: float):
        """
        :param a: a in a*x+b
        :param b: b in a*x+b
        """
        super().__init__(1)
        self._a, self._b = a, b

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        return input_data * self._a + self._b
