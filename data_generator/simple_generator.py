import data_generator.base as dg_base

import theories.base as theories_base
import environments.env_1 as env_1

import torch
from typing import Optional


class SimpleGenerator(dg_base.GeneratorBase):
    @staticmethod
    def _env_1_ask(n: int):
        return 1 + 2 * torch.rand(n)

    def ask(self, theory: theories_base.TheoryBase, previous_exploration_input: Optional[torch.tensor]) -> torch.tensor:
        """
        :param theory: Theory which is used to explore the environment.
        If ask has already been called for the generator, the theory learnt in the previous step should be passed.
        :param previous_exploration_input:  Input that has been passed to the previous exploration step.
        If no exploration has been made, this should be None.
        :return: Input for the next exploration step.
        """
        super().ask(theory, previous_exploration_input)
        new_data_size = 2 if previous_exploration_input is None else previous_exploration_input.shape[0] + 1
        if self._env.__class__ == env_1.Environment1:
            return self._env_1_ask(new_data_size)
        return 50 * torch.rand(new_data_size, self._env.parameters_count)
