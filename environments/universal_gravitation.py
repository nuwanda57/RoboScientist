import environments.base as base
import lib.success_criterion as sc
import torch


class UniversalGravityEnvironment(base.EnvironmentBase):
    """
    Environment representing 3 body gravitation.
    """
    def __init__(self):
        super().__init__(3)
        self.G = 6.67430 * 1e-11

    def run_experiments(self, input_data: torch.Tensor):
        super().run_experiments(input_data)
        return (self.G * input_data[:, 0] * input_data[:, 1] / input_data[:, 2]**2).reshape(-1, 1)
