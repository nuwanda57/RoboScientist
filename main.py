import learning_agent.robo_scientist as rs
from environments import ohm_law, single_param_linear_law, universal_gravitation, env_1
import os
from theories import theory_feynman


def main():
    feynman_dir = os.path.join(os.getcwd(), 'theories/feynman/')
    working_dirs = {
        theory_feynman.TheoryFeynman: feynman_dir
    }

    agent = rs.RoboScientist(working_dirs)
    print('\n\n------------------------------ ENV-1 ------------------------------')
    agent.explore_environment(env_1.Environment1(), theory_feynman.TheoryFeynman)
    print('Answer:', agent.get_formula_for_theory(theory_feynman.TheoryFeynman))
    # print('\n\n------------------------Single Param Linear Law-----------------------')
    # agent.explore_environment(single_param_linear_law.LinearLawEnvironment(57, 2020))

    # print('\n\n------------------------------Universal Gravity------------------------------')
    # agent.explore_environment(universal_gravitation.UniversalGravityEnvironment())


if __name__ == '__main__':
    main()
