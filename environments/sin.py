import environments.base as base
import torch


class SinEnvironment(base.EnvironmentBase):
    """
    Environment representing the Ohm's law.
    """
    def __init__(self):
        """
        :param resistance: Resistance value in the Ohm's law.
        """
        super().__init__(1)

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        return torch.sin(input_data)
