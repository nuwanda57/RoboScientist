import baselines.ai_feynman.main as baseline
from environments import ohm_law, single_param_linear_law, universal_gravitation, env_1


def main():
    agent = baseline.AIFeynman()
    # print('\n\n------------------------------Ohm\'s Law------------------------------')
    # agent.explore_environment(ohm_law.OhmLawEnvironment(2))
    # print('\n\n------------------------Single Param Linear Law-----------------------')
    # agent.explore_environment(single_param_linear_law.LinearLawEnvironment(57, 2020))

    # print('\n\n------------------------------Universal Gravity------------------------------')
    # agent.explore_environment(universal_gravitation.UniversalGravityEnvironment())

    print('\n\n------------------------------Env-1\'s Law------------------------------')
    agent.explore_environment(env_1.Environment1()).show_formula()



if __name__ == '__main__':
    main()
