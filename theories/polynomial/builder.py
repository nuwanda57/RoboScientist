import types
import functools

from theories.polynomial import polinomial_chebyshev as chebyshev
from theories.polynomial import polynomial_fourier as fourier


_INF = 100000


class NotimplementedError(Exception):
    pass


class UnknownPolynomialError(Exception):
    pass


class UnknownFunctionError(Exception):
    pass


class UnsupportedDimension(Exception):
    pass


def _copy_func(f):
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


class PolynomialBuilder(object):
    _AVAILABLE_TYPES = ['Chebyshev', 'Fourier']
    _POLYNOMIAL_HOLDER = {
        'Chebyshev': chebyshev.ChebyshevPolynomial,
        'Fourier': fourier.FourierPolynomial,
    }

    @classmethod
    def list_available(cls):
        print(cls._AVAILABLE_TYPES)


    @classmethod
    def get_polynomial_by_type(cls, ptype):
        if ptype in cls._POLYNOMIAL_HOLDER:
            return cls._POLYNOMIAL_HOLDER[ptype]
        raise UnknownPolynomialError


    @classmethod
    def get_polynomial_sets_by_type(cls, ptype):
        if ptype in cls._POLYNOMIAL_HOLDER:
            return {
                'p': PolynomialBuilder.build_polynomials_for_2d_function(ptype),
                'd_1': PolynomialBuilder.build_d_polynomials_for_2d_function(ptype, 0),
                'd_2': PolynomialBuilder.build_d_polynomials_for_2d_function(ptype, 1)
            }
        raise UnknownPolynomialError

    @classmethod
    def build_polynomials_for_2d_function(cls, ptype, cnt=None):
        if ptype not in cls._POLYNOMIAL_HOLDER:
            raise UnknownPolynomialError
        polynom_cls = cls._POLYNOMIAL_HOLDER[ptype]
        P_polynoms = []
        for deg in range(50):
            P_polynoms.append(cls._polynomial_builder_for_2d_function(deg, True, polynom_cls.p))
            t = P_polynoms[0]
            P_polynoms.append(cls._polynomial_builder_for_2d_function(deg, False, polynom_cls.p))

            for first_monom_deg in range(1, (deg + 1) // 2):
                P_polynoms.append(cls._polynomial_combination_builder_for_2d_function(
                    first_monom_deg, deg - first_monom_deg, polynom_cls.p, polynom_cls.p))
                if first_monom_deg * 2 != deg:
                    P_polynoms.append(cls._polynomial_combination_builder_for_2d_function(
                        deg - first_monom_deg, first_monom_deg, polynom_cls.p, polynom_cls.p))
        return P_polynoms

    @classmethod
    def build_d_polynomials_for_2d_function(cls, ptype, d_ind, cnt=None):
        """
        p_type: polynomial type
        d_ind: derivative index
        cnt: number of polynomials to return. Currently unsupported
        """
        if ptype not in cls._POLYNOMIAL_HOLDER:
            raise UnknownPolynomialError
        if d_ind != 0 and d_ind != 1:
            raise UnsupportedDimension
        polynom_cls = cls._POLYNOMIAL_HOLDER[ptype]
        P_polynoms = []
        for deg in range(50):
            if d_ind == 0:
                P_polynoms.append(cls._polynomial_builder_for_2d_function(deg, True, polynom_cls.d_p))
                P_polynoms.append(lambda x1, x2: 0)
            else: # d_ind == 1
                P_polynoms.append(lambda x1, x2: 0)
                P_polynoms.append(cls._polynomial_builder_for_2d_function(deg, False, polynom_cls.d_p))

            for first_monom_deg in range(1, (deg + 1) // 2):
                if d_ind == 0:
                    P_polynoms.append(cls._polynomial_combination_builder_for_2d_function(
                        first_monom_deg, deg - first_monom_deg, polynom_cls.d_p, polynom_cls.p))
                    if first_monom_deg * 2 != deg:
                        P_polynoms.append(cls._polynomial_combination_builder_for_2d_function(
                            deg - first_monom_deg, first_monom_deg, polynom_cls.d_p, polynom_cls.p))
                else:
                    P_polynoms.append(cls._polynomial_combination_builder_for_2d_function(
                        first_monom_deg, deg - first_monom_deg, polynom_cls.p, polynom_cls.d_p))
                    if first_monom_deg * 2 != deg:
                        P_polynoms.append(cls._polynomial_combination_builder_for_2d_function(
                            deg - first_monom_deg, first_monom_deg, polynom_cls.p, polynom_cls.d_p))
        return P_polynoms

    @staticmethod
    def _polynomial_builder_for_2d_function(n, is_first_arg, func):
        t = _copy_func(func)
        t.__defaults__ = (n,)
        if is_first_arg:
            return lambda x1, x2: t(x1)
        return lambda x1, x2: t(x2)

    @staticmethod
    def _polynomial_combination_builder_for_2d_function(n1, n2, func1, func2):
        t1 = _copy_func(func1)
        t1.__defaults__ = (n1,)
        t2 = _copy_func(func2)
        t2.__defaults__ = (n2,)
        return lambda x1, x2: t1(x1) * t2(x2)
