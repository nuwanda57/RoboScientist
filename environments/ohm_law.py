import environments.base as base
import lib.success_criterion as sc
import torch


class OhmLawEnvironment(base.EnvironmentBase):
    """
    Environment representing the Ohm's law.
    """
    def __init__(self, resistance: float):
        """
        :param resistance: Resistance value in the Ohm's law.
        """
        super().__init__(sc.LossSuccessCriterion(1e-1))
        self._description = base.EnvironmentParams(1)
        self._resistance = resistance
        self.parameters_count = 1

    def run_experiments(self, input_data: torch.Tensor):
        """
        :param input_data: tensor of shape (?, 1) - currents
        :return: tensor of shape (?, 1) - resulting voltages
        """
        super().run_experiments(input_data)
        self._inputs = input_data
        self._outputs = self._inputs * self._resistance
        self._experiments_finished = True
