import environments.base as base
import torch


class CosEnvironment(base.EnvironmentBase):
    def __init__(self, a=1, b=1, include_derivatives=False):
        super().__init__(1, include_derivatives=include_derivatives)
        self._a = a
        self._b = b

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        if not self._include_derivatives:
            return self._a * torch.cos(self._b * input_data)
        else:
            return torch.cat((self._a * torch.cos(self._b * input_data),
                              - self._a * self._b * torch.sin(self._b * input_data)), 0)
