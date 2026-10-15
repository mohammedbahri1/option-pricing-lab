"""Exemple d'utilisation : python examples/demo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from option_pricing import call, crr_price, greeks, implied_vol, mc_price, put  # noqa: E402

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

print(f"Call BSM            : {float(call(S, K, T, r, sigma)):.4f}")
print(f"Put BSM             : {float(put(S, K, T, r, sigma)):.4f}")

g = greeks(S, K, T, r, sigma)
print("Greeks du call      : " + ", ".join(f"{k} = {float(v):.4f}" for k, v in g.items()))

print(f"Vol. implicite      : {implied_vol(10.4506, S, K, T, r):.4%} (prix de marché 10.4506)")

print(f"Call CRR (1000 pas) : {crr_price(S, K, T, r, sigma, steps=1000):.4f}")
print(f"Put américain CRR   : {crr_price(S, K, T, r, sigma, steps=1000, kind='put', american=True):.4f}")

res = mc_price(S, K, T, r, sigma, n_paths=1_000_000, control_variate=True)
low, high = res.conf_int_95
print(f"Call Monte Carlo    : {res.price:.4f}  IC 95 % [{low:.4f} ; {high:.4f}]")
