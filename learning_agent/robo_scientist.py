import environments.base as env_base
import theories.base as theory_base
import data_generator.base as gen_base
from lib import logger as logger_config

from copy import deepcopy
from typing import Type, Dict, Optional
from collections import namedtuple
import os


EpochHistory = namedtuple('EpochHistory', ['mse', 'formula'])
ExplorationKey = namedtuple('ExplorationKey', ['env', 'theory'])


class RoboScientist(object):
    """
    The learning agent.
    """
    def __init__(self,
                 working_directories: Dict[Type[theory_base.TheoryBase], str],
                 keep_full_history: bool=False):
        """
        :param working_directories: Mapping from theory classes to working directory.
        Current directory will be changed accordingly.
        :param keep_full_history: whether to keep the full history for each (env, theory) pair or not
        """
        self._logger = logger_config.create_logger('rs')
        self._best_results = {}  # Dict[Tuple[Type[env_base.EnvironmentBase], Type[theory_base.TheoryBase]], theory_base.TheoryBase]
        self._logger.info(('Creating RoboScientist object with the following configurations:\n'
                           'working directories: {}\nkeep_full_history: {}').format(working_directories, keep_full_history))
        self._working_directories = working_directories
        self._keep_full_history = keep_full_history
        self._history = {} if self._keep_full_history else None

    def explore_environment(self, new_env: env_base.EnvironmentBase,
                            theory_class: Type[theory_base.TheoryBase],
                            generator_class: Type[gen_base.GeneratorBase],
                            epochs: int = 10) -> theory_base.TheoryBase:
        """
        :param new_env: Environment to explore.
        :param theory_class: Theory class that will be used for the exploration.
        :param generator_class: Generator class that will be used for the exploration.
        :param epochs: Number of exploration steps.
        :return: Best learnt theory.
        """
        self._logger.info('Starting {} exploration by using {} ... '.format(new_env.__class__.__name__, theory_class.__name__))
        key = ExplorationKey(env=new_env.__class__.__name__, theory=theory_class.__name__)
        self._reset_history(key)
        current_dir = os.getcwd()
        if theory_class in self._working_directories:
            os.chdir(self._working_directories[theory_class])
            self._logger.info('Current working directory: {}\n\tChanging to {} ...'.format(
                current_dir, self._working_directories[theory_class]))

        X_train = None
        generator = generator_class(new_env)
        theory = theory_class(new_env.parameters_count)
        for epoch in range(epochs):
            self._logger.info('------------------------------------------------------------------------------------\n')
            self._logger.info('EPOCH {}'.format(epoch))
            X_train = generator.ask(theory, X_train)
            y_train = new_env.run_experiments(X_train)
            theory.train(X_train, y_train)
            self._update_history(key, theory, generator, new_env)

        if theory_class in self._working_directories:
            self._logger.info('Current working directory: %s\n\tChanging to %s ...' % (os.getcwd(), current_dir))
            os.chdir(current_dir)
        return deepcopy(self._best_results[key])

    def get_formula_for_exploration_key(self, key: ExplorationKey) -> Optional[str]:
        if key in self._best_results:
            return self._best_results[key].get_formula()
        self._logger.warning(('Formula for ({1} environment, {2} theory) pair does not exist. Make sure to explore the '
                              '{1} environment by using {2} theory').format(key.env, key.theory))

    def get_theory_for_exploration_key(self, key: ExplorationKey) -> Optional[theory_base.TheoryBase]:
        if key in self._best_results:
            return self._best_results[key]
        self._logger.warning(('Theory for ({1} environment, {2} theory) pair does not exist. Make sure to explore the '
                              '{1} environment by using {2} theory').format(key.env, key.theory))

    def get_full_history(self):
        if self._keep_full_history:
            return deepcopy(self._history)
        self._logger.warning(('keep_full_history parameter is set to False: no history tracking. To start tracking '
                              'history make sure to set keep_full_history to True.'))
        return None

    def get_history_for_exploration_key(self, key):
        if self._keep_full_history:
            if key in self._history:
                return deepcopy(self._history[key])
            self._logger.warning('No history is found for ({}, {}) pair.'.format(key.env, key.theory))
            return []
        self._logger.warning(('keep_full_history parameter is set to False: no history tracking. To start tracking '
                              'history make sure to set keep_full_history to True.'))
        return []

    def _update_history(self, key, theory, generator, new_env):
        X_test = generator.ask(theory, None)
        y_test = new_env.run_experiments(X_test)
        mse = theory.calculate_test_mse(X_test, y_test)
        formula = theory.get_formula()
        self._logger.info('new MSE: {}'.format(mse))
        self._logger.info('new formula: {}'.format(formula))
        old_theory = self._best_results[key]
        if old_theory is None:
            self._logger.info('Setting best theory for ({}, {}) pair. MSE: {}, FORMULA: {}'.format(
                key.env, key.theory, mse, formula))
            self._best_results[key] = deepcopy(theory)
        else:
            old_mse = old_theory.calculate_test_mse(X_test, y_test)
            old_formula = old_theory.get_formula()
            self._logger.info('old MSE: {}'.format(old_mse))
            self._logger.info('old formula: {}'.format(old_formula))
            if old_formula == formula:
                self._logger.info(
                    ('Learnt formula is exactly the same as the previous one for ({}, {}) pair.'
                     'MSE: {}, FORMULA: {}').format(key.env, key.theory, mse, formula))
            elif old_mse < mse:
                self._logger.info(
                    ('The previous theory was better than the new one for ({}, {}) pair.'
                     'previous MSE: {}, new MSE: {}, keeping FORMULA: {},'
                     'instead of the new formula: {}').format(key.env, key.theory, old_mse, mse, old_formula, formula))
            else:
                self._logger.info(
                    ('GREAT NEWS!! The new result is better than the previous one for ({}, {}) pair.'
                     'Updating theory: previous MSE: {}, new MSE: {}, old formula: {},'
                     'NEW FORMULA: {}').format(key.env, key.theory, old_mse, mse, old_formula, formula))
                self._best_results[key] = deepcopy(theory)
        if self._keep_full_history:
            if key not in self._history:
                self._logger.critical('The key for ({}, {}) pair is not found in history.'.format(key.env, key.theory))
                raise 1
            self._history[key].append(EpochHistory(
                mse=theory.calculate_test_mse(X_test, y_test),
                formula=theory.get_formula()))
            self._logger.info(
                'Updating history for ({}, {}) pair. MSE: {}, FORMULA: {}'.format(key.env, key.theory, mse, formula))
        else:
            self._logger.info('Not tracking history.')

    def _reset_history(self, key):
        self._logger.info('Resetting history ... Clearing results for ({}, {}) pair'.format(key.env, key.theory))
        self._best_results[key] = None
        if self._keep_full_history:
            self._logger.info('Clearing full history for ({}, {}) pair'.format(key.env, key.theory))
            self._history[key] = []
        else:
            self._logger.info('Not tracking full history: nothing to clear')

