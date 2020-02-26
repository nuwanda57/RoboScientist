import environments.base as base
import lib.success_criterion as sc
import torch


class LinearLawEnvironment(base.EnvironmentBase):
    """
    Environment a*x + b.
    """
    def __init__(self, a: float, b: float):
        """
        :param a
        :param b
        """
        super().__init__(sc.LossSuccessCriterion(1))
        self._description = base.EnvironmentParams(1)
        self._a, self._b = a, b
        self.parameters_count = 1

    def run_experiments(self, input_data: torch.Tensor):
        """
        :param input_data: tensor of shape (?, 1) - currents
        :return: tensor of shape (?, 1) - resulting voltages
        """
        super().run_experiments(input_data)
        self._inputs = input_data
        self._outputs = self._inputs * self._a + self._b
        self._experiments_finished = True
