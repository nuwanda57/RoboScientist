import environments.base as env_base
import lib.theory as theory
import torch
import numpy as np

from baselines.ai_feynman.aiFeynman import aiFeynman


class AIFeynman(object):
    """
    The learning agent.
    """
    def __init__(self):
        pass

    def explore_environment(self, new_env: env_base.EnvironmentBase) -> theory.Theory:
        env_description = new_env.describe()
        X_train = new_env.get_default_input()
        y_train = new_env.run_experiments(X_train)

        file_data = np.array([X_train.numpy(), y_train.numpy()]).T

        filename = '001.a'
        np.savetxt('./data/' + filename, file_data)

        aiFeynman('./data/' + filename)
        solved_file = open("results/solutions/" + filename + '.txt')

        ans = solved_file.readlines()[0].split()[1]

        new_theory = theory.Theory(env_description.parameters_count, formula_string=ans)

        return new_theory
