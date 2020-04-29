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

    current_dir = os.getcwd()
    os.chdir(feynman_dir)
    os.system('./compile.sh')
    os.chdir(current_dir)

    agent = rs.RoboScientist(working_dirs, keep_full_history=True)
    print('\n\n------------------------------ ENV-1 ------------------------------')
    agent.explore_environment(env_1.Environment1(), theory_feynman.TheoryFeynman, simple_generator.SimpleGenerator)
    print('\nAnswer:', agent.get_formula_for_exploration_key(rs.ExplorationKey(
        env=env_1.Environment1.__name__, theory=theory_feynman.TheoryFeynman.__name__)))

    print(agent.get_full_history())


if __name__ == '__main__':
    main()
