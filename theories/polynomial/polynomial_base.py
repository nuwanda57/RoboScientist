class PolynomialBase(object):
    @staticmethod
    def p(x, n):
        """
        Calculates P_n(x) for a polynomial family P_0(x), P_1(x), ...

        :param x: Point in which the polynomial should be calculated.
        :param n: Polynomial index.
        :return: Value P_n(x)
        """
        pass

    @staticmethod
    def d_p(x, n):
        """
        Calculates d(P_n(x))/(dx) - first derivative for a polynomial family P_0(x), P_1(x), ...

        :param x: Point in which the derivative should be calculated.
        :param n: Polynomial index.
        :return: Value d(P_n(x))/(dx)
        """
        pass
