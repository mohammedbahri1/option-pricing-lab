"""Tests de cohérence : valeurs de référence, parité call-put, convergences."""
import numpy as np
import pytest

from option_pricing import call, crr_price, greeks, implied_vol, mc_price, price, put

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


def test_reference_value():
    # Valeur de référence classique : call ATM, r = 5 %, sigma = 20 %, T = 1 an
    assert call(S, K, T, r, sigma) == pytest.approx(10.450583572185565, abs=1e-10)
    assert put(S, K, T, r, sigma) == pytest.approx(5.573526022256971, abs=1e-10)


@pytest.mark.parametrize("strike", [60.0, 90.0, 100.0, 110.0, 150.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(strike, q):
    lhs = call(S, strike, T, r, sigma, q) - put(S, strike, T, r, sigma, q)
    rhs = S * np.exp(-q * T) - strike * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_vectorisation():
    strikes = np.linspace(50, 150, 101)
    vec = call(S, strikes, T, r, sigma)
    loop = np.array([call(S, k, T, r, sigma) for k in strikes])
    np.testing.assert_allclose(vec, loop, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_vs_finite_differences(kind):
    g = greeks(S, K, T, r, sigma, kind=kind)
    h = 1e-4
    f = lambda **kw: price(**{"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "kind": kind, **kw})
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-3)
    assert g["vega"] == pytest.approx((f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    # theta = -dV/dT (passage du temps)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("vol", [0.1, 0.2, 0.8])
def test_implied_vol_round_trip(strike, vol):
    p = float(call(S, strike, T, r, vol))
    assert implied_vol(p, S, strike, T, r) == pytest.approx(vol, abs=1e-7)


def test_crr_converges_to_bsm():
    assert crr_price(S, K, T, r, sigma, steps=2000) == pytest.approx(call(S, K, T, r, sigma), abs=5e-3)
    assert crr_price(S, K, T, r, sigma, steps=2000, kind="put") == pytest.approx(put(S, K, T, r, sigma), abs=5e-3)


def test_american_put_early_exercise_premium():
    euro = crr_price(S, K, T, r, sigma, steps=1000, kind="put")
    amer = crr_price(S, K, T, r, sigma, steps=1000, kind="put", american=True)
    assert amer > euro
    # Sans dividende, le call américain vaut le call européen
    assert crr_price(S, K, T, r, sigma, steps=1000, american=True) == pytest.approx(
        crr_price(S, K, T, r, sigma, steps=1000), abs=1e-10)


@pytest.mark.parametrize("cv", [False, True])
def test_monte_carlo_within_confidence_interval(cv):
    res = mc_price(S, K, T, r, sigma, n_paths=400_000, control_variate=cv, seed=7)
    exact = float(call(S, K, T, r, sigma))
    assert abs(res.price - exact) < 4 * res.std_error


def test_control_variate_reduces_variance():
    plain = mc_price(S, K, T, r, sigma, antithetic=False, seed=1)
    reduced = mc_price(S, K, T, r, sigma, antithetic=True, control_variate=True, seed=1)
    assert reduced.std_error < plain.std_error / 2
