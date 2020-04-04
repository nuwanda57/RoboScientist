import environments.base as env_base
import lib.theory as theory
import torch


class AIFeynman(object):
    """
    The learning agent.
    """
    def __init__(self):
        pass

    def explore_environment(self, new_env: env_base.EnvironmentBase) -> theory.Theory:
        env_description = new_env.describe()
        new_theory = theory.Theory(env_description.parameters_count)
        success_measurements = {
            'loss': None,
            'test_mse': None,
        }
        while not new_env.is_explored(**success_measurements):
            X_train = new_env.get_default_input()
            y_train = new_env.run_experiments(X_train)
            success_measurements['loss'] = new_theory.train(X_train, y_train)
            print('loss:', success_measurements['loss'].item())
            break

        return new_theory
