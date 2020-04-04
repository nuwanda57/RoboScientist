import environments.base as base
import lib.success_criterion as sc
import torch
import math


class Environment1(base.EnvironmentBase):
    """
    Environment representing (sqrt(2)*sqrt(exp(-x^2)))/(2*sqrt(pi)).
    """
    def __init__(self):
        super().__init__(sc.LossSuccessCriterion(1e-1))
        self._description = base.EnvironmentParams(1)
        self.parameters_count = 1

    def get_default_input(self):
        with open('data/env_1_x') as f:
            data = [float(a) for a in f.readlines()]
            return torch.FloatTensor(data)

    def run_experiments(self, input_data: torch.Tensor):
        """
        :param input_data: tensor of shape (?, 1)
        :return: tensor of shape (?, 1)
        """
        super().run_experiments(input_data)
        return torch.sqrt(2 * torch.exp(-input_data ** 2)/math.pi) / 2
