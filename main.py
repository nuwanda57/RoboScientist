import learning_agent.robo_scientist as rs
from environments import ohm_law, single_param_linear_law, universal_gravitation


def main():
    agent = rs.RoboScientist()
    print('\n\n------------------------------Ohm\'s Law------------------------------')
    agent.explore_environment(ohm_law.OhmLawEnvironment(10))
    print('\n\n------------------------Single Param Linear Law-----------------------')
    agent.explore_environment(single_param_linear_law.LinearLawEnvironment(57, 2020))

    print('\n\n------------------------------Universal Gravity------------------------------')
    agent.explore_environment(universal_gravitation.UniversalGravityEnvironment())


if __name__ == '__main__':
    main()
