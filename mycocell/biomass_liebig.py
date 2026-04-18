"""
mycocell.biomass_liebig
=======================

Alternative rate law for sub-biomass reactions based on Liebig's law of the minimum.

The classical Michaelis-Menten multiplicative rate law collapses when many
substrates are involved:
    rate = kcat × prod(C_i / (Km + C_i))
Even with 20 substrates all at saturation (C_i/(Km+C_i) ≈ 0.99), the product
collapses to 0.99^20 ≈ 0.82, and if any single one drops near Km, the product
goes to near-zero regardless of others.

Liebig's law says: growth is limited by the scarcest resource.
    rate = kcat × min(C_i / C_i_ref)

But min() is not differentiable, so solvers struggle. Soft-min alternative:
    rate = kcat / (1 + sum(C_i_ref / C_i))

Properties:
- If all C_i >> C_i_ref: sum ≈ 0, rate ≈ kcat (full speed)
- If one C_i << C_i_ref: C_i_ref/C_i dominates sum, rate drops proportionally
- If all C_i = C_i_ref: sum = N, rate = kcat / (N+1)
- Smooth and well-behaved for ODE solvers

We install this rate law via a custom BiochemNet subclass that uses different
math for the designated sub-biomass reactions.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from .simulator import BiochemNet


class LiebigBiochemNet(BiochemNet):
    """
    BiochemNet variant where designated reactions use soft-min (Liebig's law)
    kinetics instead of multiplicative Michaelis-Menten.
    
    Non-designated reactions still use standard MM.
    """
    
    def __init__(self, *args, liebig_rxn_indices: Optional[List[int]] = None,
                 liebig_refs: Optional[Dict[int, np.ndarray]] = None, **kwargs):
        """
        Args (in addition to BiochemNet's):
            liebig_rxn_indices: list of reaction indices that use soft-min
            liebig_refs: dict {rxn_idx: np.array of reference concentrations}
                         Same length as the substrate list for that reaction.
                         A precursor with C = C_ref contributes "1" to the sum.
        """
        super().__init__(*args, **kwargs)
        self.liebig_rxn_indices = set(liebig_rxn_indices or [])
        self.liebig_refs = liebig_refs or {}
    
    def compute_rates(self, C: np.ndarray) -> np.ndarray:
        """Override: use soft-min for designated reactions, MM for others."""
        C = np.maximum(C, 0.0)
        rates = np.zeros(self.n_rxns)
        
        for j in range(self.n_rxns):
            if j in self.liebig_rxn_indices:
                # Soft-min / Liebig's law:
                # rate = vmax_f / (1 + sum(C_ref_i / (C_i + epsilon)))
                sub_idx = self._sub_idx[j]
                if len(sub_idx) > 0:
                    refs = self.liebig_refs.get(j)
                    if refs is None or len(refs) != len(sub_idx):
                        refs = self._sub_km[j]  # fall back to Km as reference
                    C_subs = C[sub_idx]
                    # Clamp denominator to prevent division blowup
                    # Use a soft floor: actual concentration or a tiny positive value
                    denom = np.maximum(C_subs, 1e-9)
                    sum_ratios = np.sum(refs / denom)
                    rates[j] = self.vmax_f[j] / (1.0 + sum_ratios)
                else:
                    rates[j] = 0.0
                # No reverse term for sub-biomass reactions (always irreversible)
            else:
                # Standard MM (copied from base)
                if len(self._sub_idx[j]) > 0:
                    r_fwd = self.vmax_f[j] * np.prod(
                        C[self._sub_idx[j]] /
                        (self._sub_km[j] + C[self._sub_idx[j]] + 1e-12))
                else:
                    r_fwd = 0.0
                
                if len(self._prod_idx[j]) > 0 and self.vmax_r[j] > 0:
                    r_rev = self.vmax_r[j] * np.prod(
                        C[self._prod_idx[j]] /
                        (self._prod_km[j] + C[self._prod_idx[j]] + 1e-12))
                else:
                    r_rev = 0.0
                
                rates[j] = r_fwd - r_rev
        
        return rates
    
    def knockout(self, reaction_indices: List[int]) -> 'LiebigBiochemNet':
        """Return a new LiebigBiochemNet with given reactions set to zero."""
        new_vf = self.vmax_f.copy()
        new_vr = self.vmax_r.copy()
        new_vf[reaction_indices] = 0.0
        new_vr[reaction_indices] = 0.0
        return LiebigBiochemNet(
            self.S, new_vf, new_vr, self.km_per_rxn,
            self.met_ids, self.default_km,
            liebig_rxn_indices=list(self.liebig_rxn_indices),
            liebig_refs={k: v.copy() for k, v in self.liebig_refs.items()},
        )


def build_liebig_net_for_split(
    split_model: Dict,
    vmax_f: np.ndarray,
    vmax_r: np.ndarray,
    km_per_rxn: List[Dict[str, float]],
    C0: np.ndarray,
    default_km: float = 0.1,
    ref_scale: float = 0.1,
) -> LiebigBiochemNet:
    """
    Construct a LiebigBiochemNet from a split_model, using Liebig's law
    only for the sub-biomass reactions. Exchange reactions and metabolic
    reactions stay as MM.
    
    Args:
        ref_scale: multiplier applied to the initial concentration to set
            the reference concentration. ref_scale=0.1 means rate starts
            dropping when a precursor falls below 10% of initial.
    """
    liebig_indices = list(split_model['sub_biomass_rxn_indices'].values())
    
    # Build reference concentration array per sub-biomass reaction
    liebig_refs = {}
    for cat, rxn_idx in split_model['sub_biomass_rxn_indices'].items():
        precursor_indices = split_model['category_precursor_indices'][cat]
        # Reference: ref_scale × initial concentration (fall back to 0.1 mM if C0 is zero)
        refs = np.maximum(C0[precursor_indices] * ref_scale, 0.01)
        liebig_refs[rxn_idx] = refs
    
    return LiebigBiochemNet(
        S=split_model['S'],
        vmax_f=vmax_f, vmax_r=vmax_r,
        km_per_rxn=km_per_rxn,
        met_ids=split_model['met_ids'],
        default_km=default_km,
        liebig_rxn_indices=liebig_indices,
        liebig_refs=liebig_refs,
    )
