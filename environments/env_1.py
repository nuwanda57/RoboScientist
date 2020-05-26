import environments.base as base
import lib.success_criterion as sc
import torch
import math


class Environment1(base.EnvironmentBase):
    """
    Environment representing (sqrt(2)*sqrt(exp(-x^2)))/(2*sqrt(pi)).
    """
    def __init__(self, include_derivatives=False):
        super().__init__(1, include_derivatives=include_derivatives)

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        if not self._include_derivatives:
            return torch.sqrt(2 * torch.exp(-input_data ** 2)/math.pi) / 2
        else:
            return torch.cat(
                (torch.sqrt(2 * torch.exp(-input_data ** 2) / math.pi) / 2,
                -input_data * torch.exp(-input_data ** 2) * torch.sqrt(torch.tensor(2) / torch.exp(-input_data ** 2)) /
                2 * torch.sqrt(torch.tensor(math.pi))), 0)
