import logging
import torch


class EnvironmentBase(object):
    """
    Mystery environment, runs experiments with input_data.
    """
    def __init__(self, parameters_count: int):
        """
        :param parameters_count: Number of variables that have to be passed to experiments.
        """
        self._logger = logging.getLogger('rs.%s' % self.__class__.__name__)
        self.parameters_count = parameters_count

    def run_experiments(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        :param input_data: Input data for the experiment.
        :return: Experiment results fot the input data.
        """
        pass
