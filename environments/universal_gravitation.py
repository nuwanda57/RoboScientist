import environments.base as base
import lib.success_criterion as sc
import torch


class UniversalGravityEnvironment(base.EnvironmentBase):
    """
    Environment representing the Ohm's law.
    """
    def __init__(self):
        """
        """
        super().__init__(sc.LossSuccessCriterion(1e-1))
        self._description = base.EnvironmentParams(1)
        self.G = 6.67430 * 1e-11
        self.parameters_count = 3

    def run_experiments(self, input_data: torch.Tensor):
        """
        :param input_data: tensor of shape (?, 3) - masses of the first and the second body, and the distance between the bodies
        :return: tensor of shape (?, 1) - resulting gravitational force
        """
        super().run_experiments(input_data)
        self._inputs = input_data
        self._outputs = (self.G * self._inputs[:, 0] * self._inputs[:, 1] / self._inputs[:, 2]**2).reshape(-1, 1)
        self._experiments_finished = True
