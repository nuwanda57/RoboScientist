import numpy as np

from theories.polynomial import polynomial_base as base


class FourierPolynomial(base.PolynomialBase):
    @staticmethod
    def p(x, n):
        if n == 0:
            return 1
        if n % 2 == 0:
            return np.sin(n // 2 * x)
        return np.cos((n + 1) // 2 * x)

    @staticmethod
    def d_p(x, n):
        if n == 0:
            return 0
        if n % 2 == 0:
            return (n // 2) * np.cos(n // 2 * x)
        return -((n + 1) // 2) * np.sin((n + 1) // 2 * x)
