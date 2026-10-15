"""option_pricing : pricing d'options européennes et américaines.

Modules
-------
bsm         : formule fermée de Black-Scholes-Merton, Greeks, volatilité implicite
binomial    : arbre de Cox-Ross-Rubinstein (européen et américain)
monte_carlo : Monte Carlo avec réduction de variance
"""
from .bsm import call, greeks, implied_vol, price, put
from .binomial import crr_price
from .monte_carlo import MCResult, mc_price

__all__ = ["price", "call", "put", "greeks", "implied_vol", "crr_price", "mc_price", "MCResult"]
__version__ = "1.0.0"
