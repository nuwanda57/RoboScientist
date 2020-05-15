from typing import Optional

import torch

import data_generator.base as dg_base
import theories.theory_multiple_nested_formulas as theory_multiple_nested_formulas


class STDGenerator(dg_base.GeneratorBase):
    def ask(self, theory: theory_multiple_nested_formulas.TheoryMultipleNestedFormulas,
            previous_exploration_input: Optional[torch.tensor]) -> torch.tensor:
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
        lower_left = previous_exploration_input.min(axis=0).values
        upper_right = previous_exploration_input.max(axis=0).values
        diff = 2 * (upper_right - lower_left) + 2
        grid = torch.clamp(diff * torch.rand([10, self._env.parameters_count]) + lower_left - 0.5 * diff - 1, min=0) + 1
        standard_errors = theory.std(grid)
        index_of_the_biggest_std = standard_errors.argmax()
        next_point = grid[index_of_the_biggest_std].view(1, -1)
        return torch.cat((previous_exploration_input, next_point))
