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
        self._connected = False
        self._inputs = None
        self._outputs = None
        self._experiments_finished = None
        self._success_criterion = success_criterion
        self._description = EnvironmentParams(None)
        self.parameters_count = None

    def describe(self) -> EnvironmentParams:
        return self._description

    def is_explored(self, **success_measurements) -> bool:
        return self._success_criterion.is_satisfied(**success_measurements)

    def connect(self):
        self._inputs = []
        self._outputs = []
        self._experiments_finished = True
        self._connected = True
        self._logger.info('Connected to Environment.')

    def disconnect(self):
        self._inputs = None
        self._outputs = None
        self._connected = False
        self._experiments_finished = None
        self._logger.info('Disconnected from the Environment. Inputs and Outputs are cleared.')

    def run_experiments(self, input_data):
        if not self._connected:
            raise NoConnectionError
        self._experiments_finished = False

    def are_experiments_finished(self):
        if not self._connected:
            raise NoConnectionError
        return self._experiments_finished

    def get_experiments_results(self):
        if not self._connected:
            raise NoConnectionError
        if not self.are_experiments_finished():
            raise ExperimentsRunningError
        return self._outputs
