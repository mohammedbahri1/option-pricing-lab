"""Arbre binomial de Cox-Ross-Rubinstein (CRR), options européennes et américaines.

La récurrence arrière est vectorisée : à chaque pas, toute la colonne de
l'arbre est mise à jour en une opération NumPy, soit O(N²) opérations mais
seulement N boucles Python.
"""
from __future__ import annotations

import numpy as np

__all__ = ["crr_price"]


def crr_price(S, K, T, r, sigma, steps: int = 500, q: float = 0.0,
              kind: str = "call", american: bool = False) -> float:
    """Prix d'une option par l'arbre de Cox-Ross-Rubinstein.

    Paramètres
    ----------
    steps    : nombre de pas de l'arbre
    american : True pour autoriser l'exercice anticipé
    """
    if kind not in ("call", "put"):
        raise ValueError("kind doit valoir 'call' ou 'put'")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("Probabilité risque-neutre hors de ]0, 1[ : augmenter steps")
    disc = np.exp(-r * dt)

    # Prix du sous-jacent à maturité : S * u^(N-j) * d^j, j = 0..N
    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d ** j
    payoff = (lambda x: np.maximum(x - K, 0.0)) if kind == "call" else (lambda x: np.maximum(K - x, 0.0))
    values = payoff(ST)

    for n in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            j = np.arange(n + 1)
            values = np.maximum(values, payoff(S * u ** (n - j) * d ** j))

    return float(values[0])
