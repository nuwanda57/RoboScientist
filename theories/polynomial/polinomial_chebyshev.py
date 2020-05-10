from theories.polynomial import polynomial_base as base


class ChebyshevPolynomial(base.PolynomialBase):
    @staticmethod
    def p(x, n):
        if n == 0:
            return 1
        if n == 1:
            return x
        return 2 * x * ChebyshevPolynomial.p(x, n - 1) - ChebyshevPolynomial.p(x, n - 2)

    @staticmethod
    def d_p(x, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        return 2 * x * ChebyshevPolynomial.d_p(x, n - 1) + \
               2 * ChebyshevPolynomial.p(x, n - 1) - ChebyshevPolynomial.d_p(x, n - 2)
