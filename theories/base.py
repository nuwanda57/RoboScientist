from copy import deepcopy, copy
import logging


class TheoryBase(object):
    def __init__(self, params_cnt: int = 1):
        """
        :param params_cnt:
        """
        self._logger = logging.getLogger('rs.%s' % self.__class__.__name__)
        self._logger.info('Creating {} object with params_cnt={}'.format(self.__class__.__name__, params_cnt))
        self._params_cnt = params_cnt
        self._formula_string = ''

    def train(self, X_train, y_train):
        self._logger.info('Training theory with \n\tX={};\n\ty={}'.format(X_train, y_train))
        pass

    def calculate_test_mse(self, X_test, y_test):
        self._logger.info('Calculation test MSE for \n\tX={};\n\ty={}'.format(X_test, y_test))
        pass

    def show_formula(self):
        print(self._formula_string)

    def get_formula(self):
        return copy(self._formula_string)

    def mse(self):
        pass

    def __deepcopy__(self, memodict={}):
        new_obj = TheoryBase(self._params_cnt)
        new_obj._formula_string = self._formula_string
        return new_obj
