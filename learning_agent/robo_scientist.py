import environments.base as env_base
import theories.theory_base as theory_base
import torch
from typing import Type
from copy import deepcopy
import os


class RoboScientist(object):
    """
    The learning agent.
    """
    def __init__(self, working_directories):
        self._best_theories = {}
        self._working_directories = working_directories

    # def explore_environment(
    #         self, new_env: env_base.EnvironmentBase, theory_class: Type[theory.Theory]) -> theory.Theory:
    #     env_description = new_env.describe()
    #     new_theory = theory.Theory(env_description.parameters_count)
    #     success_measurements = {
    #         'loss': None,
    #         'test_mse': None,
    #     }
    #     while not new_env.is_explored(**success_measurements):
    #         X_train = 50 * torch.rand(1000, new_env.parameters_count)
    #         y_train = new_env.run_experiments(X_train)
    #         success_measurements['loss'] = new_theory.train(X_train, y_train)
    #         print('loss:', success_measurements['loss'].item())
    #
    #         X_test = 10 * torch.rand(1000, 3)
    #         y_test = new_env.run_experiments(X_test)
    #         success_measurements['test_mse'] = new_theory.calculate_test_mse(X_test, y_test)
    #         print('test MSE:', success_measurements['test_mse'].item())
    #         new_theory.show_model()
    #
    #     self._best_theories[theory_class.__name__] = deepcopy(new_theory)
    #     return new_theory

    def explore_environment(self, new_env: env_base.EnvironmentBase,
                            theory_class: Type[theory_base.TheoryBase]) -> theory_base.TheoryBase:
        current_dir = os.getcwd()
        os.chdir(self._working_directories[theory_class])

        env_description = new_env.describe()
        new_theory = theory_class(env_description.parameters_count)
        X_train = new_env.get_default_input()
        y_train = new_env.run_experiments(X_train)
        formula = new_theory.train(X_train, y_train)
        print(formula)

        self._best_theories[theory_class] = deepcopy(new_theory)

        os.chdir(current_dir)
        return new_theory

    def get_formula_for_theory(self, theory_class):
        if theory_class in self._best_theories:
            return self._best_theories[theory_class].get_formula()
