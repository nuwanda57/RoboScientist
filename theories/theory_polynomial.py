from copy import deepcopy, copy
from lib import logger as logger_config

from theories import base


class TheoryPolynomial(base.TheoryBase):
    def __init__(self, params_cnt: int = 1, ):
        """
        :param params_cnt:
        """
        self._logger = logger_config.create_logger(self.__class__.__name__)
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
