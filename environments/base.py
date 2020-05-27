import logging
import torch


class EnvironmentBase(object):
    """
    Mystery environment, runs experiments with input_data.
    """
    def __init__(self, parameters_count: int, include_derivatives: bool=False, left=-50, right=50):
        """
        :param parameters_count: Number of variables that have to be passed to experiments.
        :param include_derivatives: Include derivatives results
        """
        self._logger = logging.getLogger('rs.%s' % self.__class__.__name__)
        self._include_derivatives = include_derivatives
        self.parameters_count = parameters_count
        self.left = left
        self.right = right

    def run_experiments(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        :param input_data: Input data for the experiment.
        :return: Experiment results fot the input data.
        """
        pass
