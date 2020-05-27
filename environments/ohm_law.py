import environments.base as base
import torch


class OhmLawEnvironment(base.EnvironmentBase):
    """
    Environment representing the Ohm's law.
    """
    def __init__(self, resistance: float, include_derivatives=False, left=-1, right=1):
        """
        :param resistance: Resistance value in the Ohm's law.
        """
        super().__init__(1, include_derivatives, left=left, right=right)
        self._resistance = resistance

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        if not self._include_derivatives:
            return input_data * self._resistance
        return torch.cat((input_data * self._resistance, torch.ones(len(input_data)) * self._resistance), 0)
