import environments.base as base
import torch


class TgEnvironment(base.EnvironmentBase):
    def __init__(self, include_derivatives=False):
        super().__init__(1, include_derivatives=include_derivatives)

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        if not self._include_derivatives:
            return torch.sin(input_data)
        else:
            return torch.cat((torch.tan(input_data), torch.sqrt(torch.tensor(1.) / torch.cos(input_data) ** 2)), 0)
