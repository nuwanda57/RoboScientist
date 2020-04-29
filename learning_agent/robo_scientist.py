import environments.base as env_base
import theories.base as theory_base
import data_generator.base as gen_base

from typing import Type, Dict
from collections import namedtuple
from copy import deepcopy
import os
import sys

import logging


EpochHistory = namedtuple('EpochHistory', ['mse', 'formula'])
ExplorationKey = namedtuple('ExplorationKey', ['env', 'theory'])


def _create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(os.getcwd(), 'logs/robo_scientist.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class RoboScientist(object):
    """
    The learning agent.
    """
    def __init__(self,
                 working_directories: Dict[Type[theory_base.TheoryBase], str],
                 logger=None,
                 keep_full_history: bool=False):
        """
        :param working_directories: Mapping from theory classes to working directory.
        Current directory will be changed accordingly.
        :param keep_full_history: whether to keep the full history for each (env, theory) pair or not
        """
        self._logger = logger
        if self._logger is None:
            self._logger = _create_logger(self.__class__.__name__)
        self._best_results = {}  # Dict[Tuple[Type[env_base.EnvironmentBase], Type[theory_base.TheoryBase]], theory_base.TheoryBase]
        self._logger.info(('Creating RoboScientist object with the following configurations:\n'
                           'working directories: %s\nkeep_full_history: %s') % (working_directories, keep_full_history))
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
        self._logger.info(
            'Starting %s environment exploration by using %s theory ... ' % (new_env.__class__.__name__,
                                                                             theory_class.__name__))
        key = ExplorationKey(env=new_env.__class__.__name__, theory=theory_class.__name__)
        self._reset_history(key)
        current_dir = os.getcwd()
        os.chdir(self._working_directories[theory_class])
        self._logger.info(
            'Current working directory: %s\n\tChanging to %s ...' % (current_dir,
                                                                     self._working_directories[theory_class]))

        X_train = None
        generator = generator_class(new_env)
        theory = theory_class(new_env.parameters_count)
        for epoch in range(epochs):
            self._logger.info('EPOCH %d\n' % epoch)
            X_train = generator.ask(theory, X_train)
            y_train = new_env.run_experiments(X_train)
            theory.train(X_train, y_train)
            self._update_history(key, theory, generator, new_env)

        self._logger.info(
            'Current working directory: %s\n\tChanging to %s ...' % (os.getcwd(), current_dir))
        os.chdir(current_dir)
        return deepcopy(self._best_results[key])

    def get_formula_for_exploration_key(self, key: ExplorationKey) -> str:
        if key in self._best_results:
            return self._best_results[key].get_formula()
        self._logger.warning(('Formula for (%s environment, %s theory) pair does not exist. Make sure to explore the '
                              '%s environment by using %s theory') % (key.env, key.theory, key.env, key.theory))

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
            self._logger.warning('No history is found for (%s environment, %s theory) pair.' % (key.env, key.theory))
            return []
        self._logger.warning(('keep_full_history parameter is set to False: no history tracking. To start tracking '
                              'history make sure to set keep_full_history to True.'))
        return []

    def _update_history(self, key, theory, generator, new_env):
        X_test = generator.ask(theory, None)
        y_test = new_env.run_experiments(X_test)
        mse = theory.calculate_test_mse(X_test, y_test)
        formula = theory.get_formula()
        old_theory = self._best_results[key]
        if old_theory is None:
            self._logger.info(
                ('Setting best theory for (%s environment, %s theory) pair.'
                 'MSE: %f, FORMULA: %s') % (key.env, key.theory, mse, formula))
            self._best_results[key] = deepcopy(theory)
        else:
            old_mse = old_theory.calculate_test_mse(X_test, y_test)
            old_formula = old_theory.get_formula()
            if old_formula == formula:
                self._logger.info(
                    ('Learnt formula is exactly the same as the previous one for (%s environment, %s theory) pair.'
                     'MSE: %f, FORMULA: %s') % (key.env, key.theory, mse, formula))
            elif old_mse < mse:
                self._logger.info(
                    ('The previous theory was better than the new one for (%s environment, %s theory) pair.'
                     'previous MSE: %f, new MSE: %f, keeping FORMULA: %s,'
                     'instead of the new formula: %s') % (key.env, key.theory, old_mse, mse, old_formula, formula))
            else:
                self._logger.info(
                    ('GREAT NEWS!! The new result is better than the previous one for (%s environment, %s theory) pair.'
                     'Updating theory: '
                     'previous MSE: %f, new MSE: %f, old formula: %s,'
                     'NEW FORMULA: %s') % (key.env, key.theory, old_mse, mse, old_formula, formula))
                self._best_results[key] = deepcopy(theory)
        if self._keep_full_history:
            if key not in self._history:
                self._logger.critical(
                    'The key for (%s environment, %s theory) pair is not found in history dict.') % (key.env, key.theory)
                raise 1
            self._history[key].append(EpochHistory(
                mse=theory.calculate_test_mse(X_test, y_test),
                formula=theory.get_formula()))
            self._logger.info(
                ('Updating history for (%s environment, %s theory) pair.'
                 'MSE: %f, FORMULA: %s') % (key.env, key.theory, mse, formula))
        else:
            self._logger.info('Not tracking history.')

    def _reset_history(self, key):
        self._logger.info(
            'Resetting history ... Clearing results for (%s environment, %s theory) pair' % (key.env, key.theory))
        self._best_results[key] = None
        if self._keep_full_history:
            self._logger.info(
                'Clearing full history for (%s environment, %s theory) pair' % (key.env, key.theory))
            self._history[key] = []
        else:
            self._logger.info('Not tracking full history: nothing to clear')

