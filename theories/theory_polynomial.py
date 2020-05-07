from copy import deepcopy, copy
import numpy as np
from sklearn.linear_model import LinearRegression

from theories import base
from theories.polynomial import builder as polynomial_builder


class TheoryPolynomial(base.TheoryBase):
    def __init__(self, params_cnt: int = 1, polynomial_type: str = 'Chebyshev', polynomial_cnt : int = 10):
        """
        :param params_cnt:
        :param polynomial_type: Polynomial type.
        :param polynomial_cnt: Polynomial count.
        """
        super().__init__(params_cnt)
        builder = polynomial_builder.PolynomialBuilder()
        self._polynomials = builder.build_polynomials_for_2d_function(
            polynomial_type, polynomial_cnt)
        self._polynomials_d1 = builder.build_d_polynomials_for_2d_function(polynomial_type, 0, polynomial_cnt)
        self._polynomials_d2 = builder.build_d_polynomials_for_2d_function(polynomial_type, 1, polynomial_cnt)
        self._model = LinearRegression()

    def train(self, X_train, y_train):
        super().train(X_train, y_train)
        F_with_grad = np.copy(y_train)
        A_with_grad = np.array(
            [[poly(x[0], x[1]) for poly in self._polynomials] for x in X_train] + \
            [[poly(x[0], x[1]) for poly in self._polynomials_d1] for x in X_train] + \
            [[poly(x[0], x[1]) for poly in self._polynomials_d2] for x in X_train])
        self._model.fit(A_with_grad, F_with_grad)

    def calculate_test_mse(self, X_test, y_test):
        super().calculate_test_mse(X_test, y_test)
        return None

    def __deepcopy__(self, memodict={}):
        # TODO(nuwanda): implement
        pass
