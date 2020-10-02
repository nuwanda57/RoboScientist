import os

import learning_agent.robo_scientist as rs
import theories.theory_feynman as theory_feynman
import theories.theory_master as theory_master
import theories.theory_polynomial1D as theory_poly1D
import theories.theory_polynomial2D as theory_poly2D
import theories.theory_nested_formulas as theory_nested_formulas
from data_generator import simple_generator, std_generator
from environments import ohm_law, single_param_linear_law, env_1, env_2, sin, tg, arcsin, cos


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
    env = env_1.Environment1

    t = theory_feynman.TheoryFeynman
    agent.explore_environment(env(), t, simple_generator.SimpleGenerator, 5)
    print_line(env.__name__)
    print('\nAnswer:', agent.get_formula_for_exploration_key(
        rs.ExplorationKey(env=env.__name__, theory=t.__name__)))


if __name__ == '__main__':
    main()
