"""Génère les figures du README dans docs/figures/.

Usage : python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from option_pricing import call, crr_price, greeks, mc_price, price  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 130,
})

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


def sensitivities():
    """Prix en fonction de K, T, r et sigma (toutes choses égales par ailleurs)."""
    grids = {
        "K": ("Strike K", np.linspace(60, 140, 200)),
        "T": ("Maturité T (années)", np.linspace(0.01, 2, 200)),
        "r": ("Taux sans risque r", np.linspace(0, 0.10, 200)),
        "sigma": ("Volatilité σ", np.linspace(0.01, 0.6, 200)),
    }
    base = dict(S=S, K=K, T=T, r=r, sigma=sigma)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (key, (label, grid)) in zip(axes.flat, grids.items()):
        for kind, color in (("call", BLUE), ("put", ORANGE)):
            ax.plot(grid, price(**{**base, key: grid}, kind=kind), color=color, label=kind.capitalize())
        ax.set_xlabel(label)
        ax.set_ylabel("Prix")
    axes[0, 0].legend()
    fig.suptitle("Sensibilité du prix Black-Scholes-Merton aux paramètres", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "sensibilites.png")
    plt.close(fig)


def greeks_figure():
    spots = np.linspace(60, 140, 300)
    g = greeks(spots, K, T, r, sigma)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, name in zip(axes, ("delta", "gamma", "vega")):
        ax.plot(spots, g[name], color=BLUE)
        ax.set_title(name.capitalize(), color=INK)
        ax.set_xlabel("Spot S")
    fig.suptitle("Greeks d'un call européen (K = 100, T = 1 an)", color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "greeks.png", bbox_inches="tight")
    plt.close(fig)


def crr_convergence():
    steps = np.arange(10, 401)
    errors = [crr_price(S, K, T, r, sigma, steps=int(n)) - float(call(S, K, T, r, sigma)) for n in steps]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, errors, color=BLUE, linewidth=1.2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Nombre de pas de l'arbre")
    ax.set_ylabel("Prix CRR − prix BSM")
    ax.set_title("Convergence oscillante de l'arbre de Cox-Ross-Rubinstein vers Black-Scholes", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "convergence_crr.png")
    plt.close(fig)


def mc_variance_reduction():
    sizes = np.logspace(3, 6, 13).astype(int)
    configs = [("Monte Carlo standard", dict(antithetic=False), MUTED),
               ("Antithétique", dict(antithetic=True), BLUE),
               ("Antithétique + variable de contrôle", dict(antithetic=True, control_variate=True), ORANGE)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, kw, color in configs:
        se = [mc_price(S, K, T, r, sigma, n_paths=int(n), seed=3, **kw).std_error for n in sizes]
        ax.loglog(sizes, se, marker="o", markersize=4, color=color, label=label)
    ax.set_xlabel("Nombre de trajectoires")
    ax.set_ylabel("Erreur standard du prix")
    ax.set_title("Réduction de variance du Monte Carlo (call ATM)", color=INK)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "monte_carlo.png")
    plt.close(fig)


if __name__ == "__main__":
    sensitivities()
    greeks_figure()
    crr_convergence()
    mc_variance_reduction()
    print(f"Figures écrites dans {OUT}")
