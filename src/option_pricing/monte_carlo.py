"""Pricing Monte Carlo d'options européennes sous le modèle de Black-Scholes.

Deux techniques de réduction de variance sont implémentées :
- variables antithétiques (Z et -Z) ;
- variable de contrôle (le sous-jacent actualisé, d'espérance connue S·e^{-qT}).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MCResult", "mc_price"]


@dataclass
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def conf_int_95(self) -> tuple[float, float]:
        half = 1.96 * self.std_error
        return self.price - half, self.price + half


def mc_price(S, K, T, r, sigma, q: float = 0.0, kind: str = "call",
             n_paths: int = 200_000, antithetic: bool = True,
             control_variate: bool = False, seed: int | None = 42) -> MCResult:
    """Prix Monte Carlo d'un call ou d'un put européen, avec erreur standard."""
    if kind not in ("call", "put"):
        raise ValueError("kind doit valoir 'call' ou 'put'")

    rng = np.random.default_rng(seed)
    n = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(n)
    if antithetic:
        Z = np.concatenate([Z, -Z])

    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    disc = np.exp(-r * T)
    X = disc * (np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0))
    Y = disc * ST                          # E[Y] = S·e^{-qT} sous la probabilité risque-neutre

    if antithetic:
        # Les paires (Z, -Z) sont dépendantes : on travaille sur les moyennes de
        # paires, qui forment un échantillon i.i.d.
        X = 0.5 * (X[:n] + X[n:])
        Y = 0.5 * (Y[:n] + Y[n:])

    if control_variate:
        cov = np.cov(X, Y)
        beta = cov[0, 1] / cov[1, 1]
        X = X - beta * (Y - S * np.exp(-q * T))

    std_error = X.std(ddof=1) / np.sqrt(len(X))
    n_total = 2 * n if antithetic else n
    return MCResult(price=float(X.mean()), std_error=float(std_error), n_paths=n_total)
