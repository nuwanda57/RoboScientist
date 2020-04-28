from copy import deepcopy, copy


class TheoryBase(object):
    def __init__(self, params_cnt: int = 1):
        """
        :param params_cnt:
        """
        self._params_cnt = params_cnt
        self._formula_string = ''

    def train(self, X_train, y_train):
        pass

    def calculate_test_mse(self, X_test, y_test):
        pass

    def show_formula(self):
        print(self._formula_string)

    def get_formula(self):
        return copy(self._formula_string)

    def mse(self):
        pass
