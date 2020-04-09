import environments.base as base
import lib.success_criterion as sc
import torch
import math


class Environment1(base.EnvironmentBase):
    """
    Environment representing (sqrt(2)*sqrt(exp(-x^2)))/(2*sqrt(pi)).
    """
    def __init__(self):
        super().__init__(1)

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        return torch.sqrt(2 * torch.exp(-input_data ** 2)/math.pi) / 2
