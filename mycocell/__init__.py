"""
mycocell — Minimal Mycoplasma cell simulator

A multi-scale simulator coupling:
  - BiochemNet (deterministic ODE)
  - Particle dynamics (stochastic enzymes)
  - Voxel grid (spatial reaction-diffusion)

Usage:
    from mycocell import imb155, kinetics, essentiality
    from mycocell.simulator import BiochemNet, SpatialHybrid
"""

__version__ = "0.1.0"
