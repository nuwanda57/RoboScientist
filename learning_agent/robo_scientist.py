import environments.base as env_base
import lib.theory as theory
import torch


class RoboScientist(object):
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
            X_train = 50 * torch.rand(1000, new_env.parameters_count)
            y_train = new_env.run_experiments(X_train)
            success_measurements['loss'] = new_theory.train(X_train, y_train)
            print('loss:', success_measurements['loss'].item())

            X_test = 10 * torch.rand(1000, 3)
            y_test = new_env.run_experiments(X_test)
            success_measurements['test_mse'] = new_theory.calculate_test_mse(X_test, y_test)
            print('test MSE:', success_measurements['test_mse'].item())
            new_theory.show_model()

        return new_theory
