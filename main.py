import learning_agent.robo_scientist as rs
import theories.theory_feynman as theory_feynman
import theories.theory_nested_formulas as theory_nested_formulas
from data_generator import simple_generator
from environments import ohm_law, single_param_linear_law, universal_gravitation, env_1, sin

import os


def main(using="feynman"):
    feynman_dir = os.path.join(os.getcwd(), 'theories/feynman/')
    nested_formulas_dir = os.path.join(os.getcwd(), 'theories/nested_formulas/')
    working_dirs = {
        theory_feynman.TheoryFeynman: feynman_dir,
        theory_nested_formulas.TheoryNestedFormula: nested_formulas_dir
    }
   
    if using == "nested_formulas":
        theory = theory_nested_formulas.TheoryNestedFormula
    elif using == "feynman":
        theory = theory_feynman.TheoryFeynman
        current_dir = os.getcwd()
        os.chdir(feynman_dir)
        os.system('./compile.sh')
        os.chdir(current_dir)
    
    agent = rs.RoboScientist(working_dirs, keep_full_history=True)
    print('\n\n------------------------------ ENV-1 ------------------------------')
    agent.explore_environment(env_1.Environment1(), theory, simple_generator.SimpleGenerator)
    print('\nAnswer:', agent.get_formula_for_exploration_key(rs.ExplorationKey(
        env=env_1.Environment1.__name__, theory=theory.__name__)))

    print('\n\n------------------------------ OHM\'s LAW ------------------------------')
    agent.explore_environment(ohm_law.OhmLawEnvironment(1), theory,
                              simple_generator.SimpleGenerator)
    print('\nAnswer:', agent.get_formula_for_exploration_key(rs.ExplorationKey(
        env=ohm_law.OhmLawEnvironment.__name__, theory=theory.__name__)))

    print('\n\n--------------------------- SINGLE PARAM LINEAR ---------------------------')
    agent.explore_environment(single_param_linear_law.LinearLawEnvironment(2, 3), theory,
                              simple_generator.SimpleGenerator)
    print('\nAnswer:', agent.get_formula_for_exploration_key(rs.ExplorationKey(
        env=single_param_linear_law.LinearLawEnvironment.__name__, theory=theory.__name__)))

    print('\n\n-------------------------------- SIN --------------------------------')
    agent.explore_environment(sin.SinEnvironment(), theory,
                              simple_generator.SimpleGenerator)
    print('\nAnswer:', agent.get_formula_for_exploration_key(rs.ExplorationKey(
        env=sin.SinEnvironment.__name__, theory=theory.__name__)))

    d = agent.get_full_history()
    for k in d:
        print(k, '\n\t', d[k], '\n')


if __name__ == '__main__':
    main()
