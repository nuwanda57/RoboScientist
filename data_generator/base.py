import environments.base as env_base
import theories.base as theories_base

import torch
from typing import Optional, Type


class GeneratorBase(object):
    def __init__(self, environment: env_base.EnvironmentBase):
        """
        :param environment: Environment for which input should be generated.
        """
        self._env = environment

    def ask(self, theory: theories_base.TheoryBase, previous_exploration_input: Optional[torch.tensor]) -> torch.tensor:
        """
        :param theory: Theory which is used to explore the environment.
        :param previous_exploration_input:  Input that has been passed to the previous exploration step.
        If no exploration has been made, this should be None.
        :return: Input for the next exploration step.
        """
        pass