from typing import Optional

import torch

import data_generator.base as dg_base
import theories.theory_master as theory_master


class STDGenerator(dg_base.GeneratorBase):
    def __init__(self, environment, n_attempts=100):
        super(STDGenerator, self).__init__(environment)
        self.n_attempts = n_attempts

    def ask(self, theory: theory_master.MasterTheory,
            previous_exploration_input: Optional[torch.tensor], cnt=None) -> torch.tensor:
        """
        :param theory: Theory which is used to explore the environment.
        If ask has already been called for the generator, the theory learnt in the previous step should be passed.
        :param previous_exploration_input:  Input that has been passed to the previous exploration step.
        If no exploration has been made, this should be None.
        :return: Input for the next exploration step.
        """

        super().ask(theory, previous_exploration_input)

        if previous_exploration_input is None:
            return torch.rand([2, self._env.parameters_count])

        grid = torch.rand([self.n_attempts, self._env.parameters_count])

        standard_errors = theory.std(grid)
        index_of_the_biggest_std = standard_errors.argmax()
        next_point = grid[index_of_the_biggest_std].view(1, -1)
        return torch.cat((previous_exploration_input, next_point))
