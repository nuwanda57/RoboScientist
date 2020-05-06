import pytest

from theories.polynomial import polinomial_chebyshev as chebyshev


def test_polynomial_chebyshev_p():
    polynomial = chebyshev.ChebyshevPolynomial()
    assert polynomial.p(3, 0) == pytest.approx(1)
    assert polynomial.p(-1, 0) == pytest.approx(1)

    assert polynomial.p(3, 1) == pytest.approx(3)
    assert polynomial.p(-1, 1) == pytest.approx(-1)
    assert polynomial.p(157, 1) == pytest.approx(157)

    assert polynomial.p(1, 2) == pytest.approx(1)
    assert polynomial.p(-1, 2) == pytest.approx(1)
    assert polynomial.p(7, 2) == pytest.approx(97)

    assert polynomial.p(0.1, 3) == pytest.approx(-0.296)
    assert polynomial.p(-1.17, 3) == pytest.approx(-2.89645)
    assert polynomial.p(7, 3) == pytest.approx(1351)

    assert polynomial.p(1.1, 10) == pytest.approx(42.2108)
    assert polynomial.p(-0.5, 10) == pytest.approx(-0.5)
    assert polynomial.p(-0.23, 10) == pytest.approx(0.681624)


def test_polynomial_chebyshev_dp():
    polynomial = chebyshev.ChebyshevPolynomial()
    assert polynomial.d_p(3, 0) == pytest.approx(0)
    assert polynomial.d_p(-1, 0) == pytest.approx(0)

    assert polynomial.d_p(3, 1) == pytest.approx(1)
    assert polynomial.d_p(-1, 1) == pytest.approx(1)
    assert polynomial.d_p(157, 1) == pytest.approx(1)

    assert polynomial.d_p(1, 2) == pytest.approx(4)
    assert polynomial.d_p(-1, 2) == pytest.approx(-4)
    assert polynomial.d_p(7, 2) == pytest.approx(28)

    assert polynomial.d_p(0.1, 3) == pytest.approx(-2.88)
    assert polynomial.d_p(-1.17, 3) == pytest.approx(13.4268)
    assert polynomial.d_p(7, 3) == pytest.approx(585)

    assert polynomial.d_p(1.1, 10) == pytest.approx(920.856)
    assert polynomial.d_p(-0.5, 10) == pytest.approx(10)
    assert polynomial.d_p(-0.23, 10) == pytest.approx(0-7.51859)
