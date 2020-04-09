import learning_agent.robo_scientist as rs
import theories.theory_feynman as theory_feynman
from data_generator import simple_generator
from environments import ohm_law, single_param_linear_law, universal_gravitation, env_1

import os


def main():
    feynman_dir = os.path.join(os.getcwd(), 'theories/feynman/')
    working_dirs = {
        theory_feynman.TheoryFeynman: feynman_dir
    }

    agent = rs.RoboScientist(working_dirs)
    print('\n\n------------------------------ ENV-1 ------------------------------')
    agent.explore_environment(env_1.Environment1(), theory_feynman.TheoryFeynman, simple_generator.SimpleGenerator)
    print('\nAnswer:', agent.get_formula_for_theory(theory_feynman.TheoryFeynman))


if __name__ == '__main__':
    main()
