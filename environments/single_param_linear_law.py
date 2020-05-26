import environments.base as base
import lib.success_criterion as sc
import torch


class LinearLawEnvironment(base.EnvironmentBase):
    """
    Environment a*x + b.
    """
    def __init__(self, a: float, b: float, include_derivatives=False):
        """
        :param a: a in a*x+b
        :param b: b in a*x+b
        """
        super().__init__(1, include_derivatives=include_derivatives)
        self._a, self._b = a, b

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        if not self._include_derivatives:
            return input_data * self._a + self._b
        else:
            return torch.cat((input_data * self._a + self._b, torch.ones(len(input_data)) * self._a), 0)
