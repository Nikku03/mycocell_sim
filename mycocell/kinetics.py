"""
mycocell.kinetics
=================

Kinetic parameters for iMB155 reactions.

Sources:
  - 10 reactions: literature-derived (BRENDA, Breuer 2019)
  - 234 reactions: default values (kcat=10 mM/s, Km=0.1 mM)

For production use, the default values should be replaced with measured
values from Thornburg 2022 Table S3 or learned via fitting to trajectory data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EnzymeKinetics:
    vmax: float                    # max velocity, mM/s
    km: Dict[str, float]           # Michaelis constants per substrate, mM
    ki: Dict[str, float] = field(default_factory=dict)
    ka: Dict[str, float] = field(default_factory=dict)
    hill: float = 1.0
    reversible: bool = True
    keq: float = 1.0


# ---------------------------------------------------------------
# Literature-derived kinetic parameters (15 key glycolysis reactions)
# From Breuer 2019 iMB155 paper, BRENDA, and EcoCyc
# ---------------------------------------------------------------

LITERATURE_KINETICS: Dict[str, EnzymeKinetics] = {
    'GLCpts': EnzymeKinetics(
        vmax=10.0, km={'glc': 0.05, 'pep': 0.1},
        ki={'g6p': 1.0}, reversible=False),
    'PGI': EnzymeKinetics(
        vmax=100.0, km={'g6p': 0.5, 'f6p': 0.2},
        keq=0.4, reversible=True),
    'PFK': EnzymeKinetics(
        vmax=50.0, km={'f6p': 0.1, 'atp': 0.05},
        ki={'atp': 2.0, 'pep': 0.5}, ka={'adp': 0.5, 'amp': 0.1},
        hill=4.0, reversible=False),
    'FBA': EnzymeKinetics(
        vmax=50.0, km={'fbp': 0.05, 'g3p': 0.1, 'dhap': 0.1},
        keq=0.1, reversible=True),
    'TPI': EnzymeKinetics(
        vmax=500.0, km={'dhap': 0.5, 'g3p': 0.5},
        keq=0.05, reversible=True),
    'GAPD': EnzymeKinetics(
        vmax=100.0, km={'g3p': 0.05, 'nad': 0.1, 'pi': 1.0,
                        'bpg13': 0.01, 'nadh': 0.01},
        keq=0.01, reversible=True),
    'PGK': EnzymeKinetics(
        vmax=200.0, km={'bpg13': 0.01, 'adp': 0.1, 'pg3': 0.5, 'atp': 0.5},
        keq=3000, reversible=True),
    'PGM': EnzymeKinetics(
        vmax=100.0, km={'pg3': 0.2, 'pg2': 0.1},
        keq=0.15, reversible=True),
    'ENO': EnzymeKinetics(
        vmax=100.0, km={'pg2': 0.2, 'pep': 0.1, 'h2o': 1.0},
        keq=6.0, reversible=True),
    'PYK': EnzymeKinetics(
        vmax=100.0, km={'pep': 0.1, 'adp': 0.1, 'pyr': 1.0, 'atp': 1.0},
        keq=1e4, reversible=False),
    'LDH': EnzymeKinetics(
        vmax=200.0, km={'pyr': 0.5, 'nadh': 0.02, 'lac': 5.0, 'nad': 0.5},
        keq=1e4, reversible=True),
    'G6PDH': EnzymeKinetics(
        vmax=20.0, km={'g6p': 0.3, 'nadp': 0.01, '6pg': 0.5, 'nadph': 0.01},
        keq=5.0, reversible=True),
    'GND': EnzymeKinetics(
        vmax=30.0, km={'6pg': 0.1, 'nadp': 0.02, 'ru5p': 0.1, 'nadph': 0.02, 'co2': 1.0},
        keq=100, reversible=True),
    'ADK': EnzymeKinetics(
        vmax=500.0, km={'atp': 0.5, 'amp': 0.1, 'adp': 0.5},
        keq=1.0, reversible=True),
    'NDK': EnzymeKinetics(
        vmax=1000.0, km={'atp': 0.5, 'gdp': 0.5, 'adp': 0.5, 'gtp': 0.5},
        keq=1.0, reversible=True),
}


# Default values used when reaction-specific kinetics unavailable
DEFAULT_VMAX_F = 10.0    # mM/s
DEFAULT_VMAX_R = 1.0     # mM/s (weaker default reverse)
DEFAULT_KM = 0.1         # mM


def build_rate_arrays(rxn_ids: List[str],
                       literature: Dict[str, EnzymeKinetics] = None) -> Dict:
    """Build vmax_f, vmax_r, km_per_rxn arrays aligned with rxn_ids."""
    if literature is None:
        literature = LITERATURE_KINETICS
    
    n = len(rxn_ids)
    vmax_f = np.full(n, DEFAULT_VMAX_F)
    vmax_r = np.full(n, DEFAULT_VMAX_R)
    km_per_rxn: List[Dict[str, float]] = [{} for _ in range(n)]
    is_measured = np.zeros(n, dtype=bool)
    
    for i, rid in enumerate(rxn_ids):
        # Strip R_ prefix for lookup
        stripped = rid[2:] if rid.startswith('R_') else rid
        # Also try without _L/_D suffix
        candidates = [stripped, stripped.rstrip('_L').rstrip('_D')]
        if stripped.endswith('_L') or stripped.endswith('_D'):
            candidates.append(stripped[:-2])
        
        for cand in candidates:
            if cand in literature:
                ek = literature[cand]
                vmax_f[i] = ek.vmax
                vmax_r[i] = ek.vmax / max(ek.keq, 0.01) if ek.reversible else 0.0
                km_per_rxn[i] = dict(ek.km)
                is_measured[i] = True
                break
    
    return {
        'vmax_f': vmax_f, 'vmax_r': vmax_r,
        'km_per_rxn': km_per_rxn,
        'is_measured': is_measured,
        'default_km': DEFAULT_KM,
        'n_measured': int(is_measured.sum()),
        'n_default': int((~is_measured).sum()),
    }
