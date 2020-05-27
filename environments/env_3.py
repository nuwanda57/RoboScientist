import environments.base as base
import torch


class Environment3(base.EnvironmentBase):
    def __init__(self, include_derivatives=False):
        super().__init__(2, include_derivatives=include_derivatives, left=-1, right=1)

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        x1 = input_data[:,0]
        x2 = input_data[:,1]
        if not self._include_derivatives:
            return x1 * torch.sin(3 * x2)
        else:
            f = x1 * torch.sin(3 * x2)
            d1 = torch.sin(3 * x2)
            d2 = 3 * x1 * torch.cos(3 * x2)

            return torch.cat((f, d1, d2), 0)
