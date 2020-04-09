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
        super().__init__(1)
        self._resistance = resistance

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        return input_data * self._resistance
