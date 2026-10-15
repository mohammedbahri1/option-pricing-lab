"""Black-Scholes-Merton : prix, Greeks et volatilité implicite (options européennes).

Toutes les fonctions sont vectorisées : chaque argument peut être un scalaire
ou un tableau NumPy, avec diffusion (broadcasting) automatique.

Notations
---------
S     : prix du sous-jacent
K     : strike
T     : maturité résiduelle en années (T > 0)
r     : taux sans risque continu
sigma : volatilité annuelle
q     : taux de dividende continu (0 par défaut)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr  # fonction de répartition de la loi normale, en C

__all__ = ["d1_d2", "price", "call", "put", "greeks", "implied_vol"]


def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError("kind doit valoir 'call' ou 'put'")


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Renvoie (d1, d2) de la formule de Black-Scholes-Merton."""
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(S, K, T, r, sigma, q=0.0, kind: str = "call"):
    """Prix d'une option européenne (call ou put)."""
    _check_kind(kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * np.asarray(T)), np.exp(-q * np.asarray(T))
    if kind == "call":
        return S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    return K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)


def call(S, K, T, r, sigma, q=0.0):
    return price(S, K, T, r, sigma, q, "call")


def put(S, K, T, r, sigma, q=0.0):
    return price(S, K, T, r, sigma, q, "put")


def greeks(S, K, T, r, sigma, q=0.0, kind: str = "call") -> dict:
    """Greeks analytiques : delta, gamma, vega, theta (par an) et rho.

    Vega et rho sont exprimés pour une variation de 1 (et non 1 %).
    """
    _check_kind(kind)
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf_d1 = _norm_pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * df_q * pdf_d1 * np.sqrt(T)
    decay = -S * df_q * pdf_d1 * sigma / (2.0 * np.sqrt(T))

    if kind == "call":
        delta = df_q * ndtr(d1)
        theta = decay - r * K * df_r * ndtr(d2) + q * S * df_q * ndtr(d1)
        rho = K * T * df_r * ndtr(d2)
    else:
        delta = -df_q * ndtr(-d1)
        theta = decay + r * K * df_r * ndtr(-d2) - q * S * df_q * ndtr(-d1)
        rho = -K * T * df_r * ndtr(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(market_price, S, K, T, r, q=0.0, kind: str = "call",
                tol: float = 1e-10, max_iter: int = 50) -> float:
    """Volatilité implicite par Newton-Raphson, avec repli sur Brent.

    Newton converge en quelques itérations près de la monnaie ; Brent prend le
    relais quand la vega est trop faible (options très en dehors de la monnaie).
    """
    _check_kind(kind)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    lower = max(S * df_q - K * df_r, 0.0) if kind == "call" else max(K * df_r - S * df_q, 0.0)
    upper = S * df_q if kind == "call" else K * df_r
    if not lower <= market_price < upper:
        raise ValueError("Prix hors des bornes de non-arbitrage")

    sigma = 0.2
    for _ in range(max_iter):
        diff = float(price(S, K, T, r, sigma, q, kind)) - market_price
        if abs(diff) < tol:
            return sigma
        vega = float(greeks(S, K, T, r, sigma, q, kind)["vega"])
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if not 1e-6 < sigma < 5.0:
            break

    return brentq(lambda s: float(price(S, K, T, r, s, q, kind)) - market_price,
                  1e-6, 5.0, xtol=tol)
