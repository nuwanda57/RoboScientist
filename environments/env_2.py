import environments.base as base
import torch


class Environment2(base.EnvironmentBase):
    def __init__(self, include_derivatives=False):
        super().__init__(2, include_derivatives=include_derivatives, left=-2, right=2)

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        x1 = input_data[:,0]
        x2 = input_data[:,1]
        if not self._include_derivatives:
            return torch.exp(-torch.pow(x1, 2) - 0.5 * (x2 - 1) * x2)
        else:
            f = torch.exp(-torch.pow(x1, 2) - 0.5 * (x2 - 1) * x2)
            d1 = torch.exp(-torch.pow(x1, 2) - 0.5 * (x2 - 1) * x2) * (-2*x1)
            d2 = torch.exp(-torch.pow(x1, 2) - 0.5 * (x2 - 1) * x2) * (-x2 + 0.5)

            return torch.cat((f, d1, d2), 0)
