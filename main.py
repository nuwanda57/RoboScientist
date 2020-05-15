import os

import learning_agent.robo_scientist as rs
import theories.theory_feynman as theory_feynman
import theories.theory_multiple_nested_formulas as theory_multiple_nested_formulas
import theories.theory_nested_formulas as theory_nested_formulas
from data_generator import simple_generator, std_generator
from environments import ohm_law, single_param_linear_law, env_1, sin


def print_line(name):
    overall_length = 60
    left_part = (overall_length - len(name)) // 2
    right_part = overall_length - len(name) - left_part
    print("".join(["\n\n", "-" * left_part, name, "-" * right_part, "\n"]))


def main():
    feynman_dir = os.path.join(os.getcwd(), 'theories/feynman/')
    working_dirs = {
        theory_feynman.TheoryFeynman: feynman_dir,
    }

    current_dir = os.getcwd()
    os.chdir(feynman_dir)
    os.system('./compile.sh')
    os.chdir(current_dir)

    agent = rs.RoboScientist(working_dirs, keep_full_history=True)
    for theory, generator in zip(
        [
            theory_multiple_nested_formulas.TheoryMultipleNestedFormulas,
            theory_feynman.TheoryFeynman,
            theory_nested_formulas.TheoryNestedFormula
        ],
        [
            std_generator.STDGenerator,
            simple_generator.SimpleGenerator,
            simple_generator.SimpleGenerator,
        ]):
        print_line(theory.__name__)
        for environment, params in zip(
                [
                    env_1.Environment1,
                    ohm_law.OhmLawEnvironment,
                    single_param_linear_law.LinearLawEnvironment,
                    sin.SinEnvironment
                ],
                [None, [1], [2, 3], None]
        ):
            print_line(environment.__name__)
            if params is not None:
                agent.explore_environment(environment(*params), theory, generator)
            else:
                agent.explore_environment(environment(), theory, generator)
            print('\nAnswer:', agent.get_formula_for_exploration_key(rs.ExplorationKey(
                env=environment.__name__, theory=theory.__name__)))
    d = agent.get_full_history()
    for k in d:
        print(k, '\n\t', d[k], '\n')

if __name__ == '__main__':
    main()
