import logging
import lib.success_criterion as sc
from collections import namedtuple


class UnimplementedError(Exception):
    """Raised when no Implementation exists."""
    pass


class NoConnectionError(Exception):
    """No connection with Environment. Make sure to connect before running experiments."""
    pass


class ExperimentsRunningError(Exception):
    """Experiments are still running. Make sure experiments are finished before collecting the results."""
    pass


EnvironmentParams = namedtuple('EnvironmentParams', ['parameters_count'])


class EnvironmentBase(object):
    """
    Base class for Environments.
    Defines protocol between the environments and the learning agent.
    """
    def __init__(self, success_criterion: sc.BaseSuccessCriterion):
        self._logger = logging.getLogger('Environment logger')
        self._success_criterion = success_criterion
        self._description = EnvironmentParams(None)
        self.parameters_count = None

    def describe(self) -> EnvironmentParams:
        return self._description

    def get_default_input(self):
        return None

    def is_explored(self, **success_measurements) -> bool:
        return self._success_criterion.is_satisfied(**success_measurements)

    def run_experiments(self, input_data):
        return []
