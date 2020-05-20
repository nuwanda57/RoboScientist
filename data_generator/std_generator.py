from typing import Optional

import torch

import data_generator.base as dg_base
import theories.theory_master as theory_master


class STDGenerator(dg_base.GeneratorBase):
    def ask(self, theory: theory_master.MasterTheory,
            previous_exploration_input: Optional[torch.tensor]) -> torch.tensor:
        """
        :param theory: Theory which is used to explore the environment.
        If ask has already been called for the generator, the theory learnt in the previous step should be passed.
        :param previous_exploration_input:  Input that has been passed to the previous exploration step.
        If no exploration has been made, this should be None.
        :return: Input for the next exploration step.
        """

        super().ask(theory, previous_exploration_input)
        self.n_samples = 10

        if previous_exploration_input is None:
            return torch.rand([2, self._env.parameters_count])

        # We sample self.n_samples points from a hypercube uniformly at random
        # Firstly, we identify the smallest and the biggest values of each variable
        lower_left = previous_exploration_input.min(axis=0).values
        upper_right = previous_exploration_input.max(axis=0).values

        # Secondly, we compute the vector that would correspond to the main diagonal of our hypercube
        # We do it in order to double in length the main diagonal of a hypercube
        # that would have lower_left and upper_right as its vertices
        diff = 2 * (upper_right - lower_left) + 2

        # Thirdly, we perform random sampling from the hypercube that is clamped
        # so that all its points have all their coordinates positive
        grid = torch.clamp(diff * torch.rand([self.n_samples, self._env.parameters_count]) + lower_left - 0.5 * diff - 1, min=0) + 1

        # Lastly, we identify the point where our theory is most incertain
        standard_errors = theory.std(grid)
        index_of_the_biggest_std = standard_errors.argmax()
        next_point = grid[index_of_the_biggest_std].view(1, -1)
        return torch.cat((previous_exploration_input, next_point))
