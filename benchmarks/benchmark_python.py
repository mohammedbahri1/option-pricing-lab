"""Benchmark : de l'implémentation « naïve » à l'implémentation vectorisée.

Point de départ : la version courante dans les supports de cours, où la loi
normale N(d) est calculée par intégration numérique (scipy.integrate.quad) et
où l'on boucle en Python sur chaque strike.

Usage : python benchmarks/benchmark_python.py
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from option_pricing import call  # noqa: E402

S, T, r, sigma = 100.0, 1.0, 0.05, 0.2
N_OPTIONS = 100_000
strikes = np.linspace(50.0, 150.0, N_OPTIONS)


# --- Version 1 : N(d) par intégration numérique, boucle Python -------------
def n_quad(d):
    return quad(lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi), -20, d, limit=50)[0]


def call_quad(St, K):
    d1 = (math.log(St / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return St * n_quad(d1) - math.exp(-r * T) * K * n_quad(d2)


# --- Version 2 : N(d) par fonction d'erreur, boucle Python -----------------
def n_erf(d):
    return 0.5 * (1.0 + math.erf(d / math.sqrt(2.0)))


def call_erf(St, K):
    d1 = (math.log(St / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return St * n_erf(d1) - math.exp(-r * T) * K * n_erf(d2)


def timeit(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    # La version quad est très lente : on la mesure sur un échantillon puis on extrapole
    sample = 2_000
    t_quad, v_quad = timeit(lambda: [call_quad(S, k) for k in strikes[:sample]], repeat=1)
    t_quad *= N_OPTIONS / sample
    t_erf, _ = timeit(lambda: [call_erf(S, k) for k in strikes])
    t_vec, v_vec = timeit(lambda: call(S, strikes, T, r, sigma))

    err = np.max(np.abs(np.array(v_quad) - v_vec[:sample]))
    print(f"Pricing de {N_OPTIONS:,} calls européens (S=100, T=1, r=5 %, sigma=20 %)\n")
    print(f"{'Implémentation':<45}{'Temps':>12}{'Accélération':>15}")
    for name, t in [("1. Boucle Python + N(d) par quad", t_quad),
                    ("2. Boucle Python + N(d) par math.erf", t_erf),
                    ("3. NumPy vectorisé + scipy.special.ndtr", t_vec)]:
        print(f"{name:<45}{t * 1e3:>10.1f} ms{t_quad / t:>14.0f}x")
    print(f"\nÉcart maximal entre versions 1 et 3 : {err:.2e}")


if __name__ == "__main__":
    main()
