from copy import deepcopy, copy
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import mean_squared_error

from theories import base
from theories.polynomial import builder as polynomial_builder


class TheoryPolynomial1D(base.TheoryBase):
    def __init__(self, params_cnt: int = 1, polynomial_type: str = 'Chebyshev', polynomial_cnt : int = 4):
        """
        :param params_cnt:
        :param polynomial_type: Polynomial type.
        :param polynomial_cnt: Polynomial count.
        """
        super().__init__(params_cnt)
        builder = polynomial_builder.PolynomialBuilder()
        self._polynomials = builder.build_polynomials_for_1d_function(
            polynomial_type, polynomial_cnt)
        self._polynomials_d = builder.build_d_polynomials_for_1d_function(polynomial_type, polynomial_cnt)
        self._model = LinearRegression()

    def train(self, X_train, y_train):
        super().train(X_train, y_train)
        F_with_grad = np.copy(y_train)
        A_with_grad = np.array(
            [[poly(x) for poly in self._polynomials] for x in X_train] + \
            [[poly(x) for poly in self._polynomials_d] for x in X_train])
        self._model.fit(A_with_grad, F_with_grad)
        self._formula_string = ["%.2f" % a for a in [self._model.intercept_] + self._model.coef_]

    def calculate_test_mse(self, X_test, y_test):
        super().calculate_test_mse(X_test, y_test)
        F_with_grad = np.copy(y_test)
        A_with_grad = np.array(
            [[poly(x) for poly in self._polynomials] for x in X_test] + \
            [[poly(x) for poly in self._polynomials_d] for x in X_test])
        return mean_squared_error(
            self._model.predict(A_with_grad), F_with_grad)

    def __deepcopy__(self, memodict={}):
        new_obj = super().__deepcopy__(memodict)
        new_obj._model = pickle.loads(pickle.dumps(self._model))
        new_obj._polynomials = deepcopy(self._polynomials)
        new_obj._polynomials_d = deepcopy(self._polynomials_d)
        return new_obj
