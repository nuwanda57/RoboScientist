import environments.base as env_base
import theories.base as theory_base
import data_generator.base as gen_base

from typing import Type, Dict
from copy import deepcopy
import os


class RoboScientist(object):
    """
    The learning agent.
    """
    def __init__(self, working_directories: Dict[Type[theory_base.TheoryBase], str]):
        self._best_theories = {}
        self._working_directories = working_directories

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
        current_dir = os.getcwd()
        os.chdir(self._working_directories[theory_class])

        X_train = None
        generator = generator_class(new_env)
        theory = theory_class(new_env.parameters_count)
        formulas = []
        for epoch in range(epochs):
            X_train = generator.ask(theory, X_train)
            y_train = new_env.run_experiments(X_train)
            theory.train(X_train, y_train)
            # TODO(nuwanda): add condition based on MSE here
            self._best_theories[theory.__class__] = deepcopy(theory)
            formulas.append(deepcopy(theory.get_formula()))

        os.chdir(current_dir)
        for f in formulas:
            print('formula:', f)
        return deepcopy(self._best_theories[theory.__class__])

    def get_formula_for_theory(self, theory_class: Type[theory_base.TheoryBase]) -> str:
        if theory_class in self._best_theories:
            return self._best_theories[theory_class].get_formula()
