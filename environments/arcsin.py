import environments.base as base
import torch


class ArcsinEnvironment(base.EnvironmentBase):
    """
    Environment representing the Ohm's law.
    """
    def __init__(self, include_derivatives=False):
        """
        :param resistance: Resistance value in the Ohm's law.
        """
        super().__init__(1, include_derivatives=include_derivatives)

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        if not self._include_derivatives:
            return torch.asin(input_data)
        else:
            return torch.cat((torch.asin(input_data), torch.sqrt(torch.tensor(1.) - input_data ** 2) ** (-1)), 0)
