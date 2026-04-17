"""
mycocell.biomass
================

Handles the biomass reaction from Syn3A_updated.xml and provides a
growth-rate-based viability metric for essentiality prediction.

Biology:
  - R_BIOMASS consumes ~53 precursors → produces 1 M_biomass_c
  - R_EX_biomass_c consumes M_biomass_c → drains (growth sink)
  - Combined effect: metabolites pulled out of system at biomass rate

Our approach:
  - Treat R_BIOMASS as Michaelis-Menten: kcat × product(C/(Km+C)) over all
    precursors. If any precursor runs low, biomass rate drops.
  - Treat R_EX_biomass_c as fast first-order: k × [biomass_c]. Keeps biomass
    pool small, so all mass pulled from R_BIOMASS flux is effectively the
    growth sink.

Viability metric:
  - Compute biomass flux averaged over last 20% of simulation
  - Knockout is "essential" if growth rate drops below threshold fraction
    of WT growth rate
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple


# Default parameters for the biomass reaction treated as MM kinetics
DEFAULT_BIOMASS_KCAT = 0.0002      # 1/s — roughly corresponds to doubling time ~1 hour
DEFAULT_BIOMASS_KM = 0.01           # mM — low Km so that the reaction proceeds
                                    # at near vmax unless precursors actually run low
DEFAULT_EXCHANGE_RATE = 100.0       # 1/s — first-order rate for M_biomass_c drainage


def configure_biomass_kinetics(
    rxn_ids: List[str],
    biomass_rxn_idx: int,
    biomass_exchange_idx: int,
    vmax_f: np.ndarray,
    vmax_r: np.ndarray,
    km_per_rxn: List[Dict[str, float]],
    met_ids: List[str],
    S: np.ndarray,
    biomass_kcat: float = DEFAULT_BIOMASS_KCAT,
    biomass_km: float = DEFAULT_BIOMASS_KM,
    exchange_rate: float = DEFAULT_EXCHANGE_RATE,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Configure the biomass and exchange reactions with MM kinetics.
    
    Modifies vmax_f, vmax_r, km_per_rxn in place and returns them.
    """
    # Biomass reaction: forward at biomass_kcat, irreversible
    vmax_f[biomass_rxn_idx] = biomass_kcat
    vmax_r[biomass_rxn_idx] = 0.0
    
    # Km for each precursor — use the same biomass_km for all
    bm_col = S[:, biomass_rxn_idx]
    precursor_indices = np.where(bm_col < 0)[0]
    km_per_rxn[biomass_rxn_idx] = {}
    for i in precursor_indices:
        met_id = met_ids[i]
        normalized = _normalize_met(met_id)
        km_per_rxn[biomass_rxn_idx][normalized] = biomass_km
    
    # Exchange reaction: first-order on [M_biomass_c]
    # Vmax * (C / (Km + C)) ≈ Vmax * C/Km when C << Km
    # We want effective rate ≈ exchange_rate * C, so set Vmax = exchange_rate * Km
    ex_col = S[:, biomass_exchange_idx]
    substrate_indices = np.where(ex_col < 0)[0]
    km_per_rxn[biomass_exchange_idx] = {}
    # Use larger Km so that effective behavior is first-order unless biomass_c
    # actually builds up
    exchange_km = 10.0  # mM
    vmax_f[biomass_exchange_idx] = exchange_rate * exchange_km
    vmax_r[biomass_exchange_idx] = 0.0
    for i in substrate_indices:
        met_id = met_ids[i]
        normalized = _normalize_met(met_id)
        km_per_rxn[biomass_exchange_idx][normalized] = exchange_km
    
    if verbose:
        print(f"  Biomass reaction (idx {biomass_rxn_idx}):")
        print(f"    vmax = {biomass_kcat} 1/s, Km = {biomass_km} mM for {len(precursor_indices)} precursors")
        print(f"    Max possible flux: {biomass_kcat:.2e} mM/s")
        print(f"  Biomass exchange (idx {biomass_exchange_idx}):")
        print(f"    vmax = {vmax_f[biomass_exchange_idx]:.1f} mM/s, Km = {exchange_km} mM")
        print(f"    Effective first-order rate: {exchange_rate} 1/s")
    
    return vmax_f, vmax_r, km_per_rxn


def _normalize_met(mid: str) -> str:
    """Strip M_ prefix and compartment suffix."""
    s = mid
    if s.startswith('M_'):
        s = s[2:]
    # Handle both _c (single underscore) and __L_c (double underscore for L/D forms)
    # We want to strip ONLY the compartment suffix, not the isomer designator
    if s.endswith('_c') or s.endswith('_e') or s.endswith('_p'):
        s = s[:-2]
    return s


def compute_biomass_flux(biochem_net, sol, biomass_rxn_idx: int,
                         last_fraction: float = 0.2) -> Dict:
    """
    Compute the biomass reaction flux over the last portion of a trajectory.
    
    Returns:
      {
        'mean_flux': average biomass flux over last fraction (mM/s),
        'final_flux': biomass flux at t=t_max (mM/s),
        'flux_trajectory': array of biomass flux at each recorded time point,
      }
    """
    # We need to compute rate at each time point
    fluxes = np.zeros(sol.t.size)
    for t_idx in range(sol.t.size):
        C = sol.y[:, t_idx]
        rates = biochem_net.compute_rates(C)
        fluxes[t_idx] = rates[biomass_rxn_idx]
    
    # Average over last portion
    n = sol.t.size
    start = int(n * (1 - last_fraction))
    mean_flux = float(fluxes[start:].mean())
    final_flux = float(fluxes[-1])
    
    return {
        'mean_flux': mean_flux,
        'final_flux': final_flux,
        'flux_trajectory': fluxes,
        't_trajectory': sol.t,
    }


def growth_rate_viability(
    biochem_net,
    C0: np.ndarray,
    biomass_rxn_idx: int,
    gene_to_rxns: Dict[str, List[int]],
    labels: Dict[str, str],
    t_max: float = 10.0,
    threshold_fraction: float = 0.1,
    verbose: bool = True,
) -> Dict:
    """
    For each gene, knockout its reactions, simulate, and measure growth rate.
    A knockout is "essential" if growth rate < threshold_fraction * WT growth.
    
    Returns:
      {
        'wt_growth_rate': WT biomass flux (mM/s),
        'threshold': threshold_fraction * wt_growth_rate,
        'results': [{gene, true_label, true_essential, ko_growth_rate,
                     predicted_essential, correct}, ...],
        'wt_sol': scipy solve_ivp result for WT,
      }
    """
    # WT baseline
    if verbose:
        print(f"  Running WT baseline ({t_max}s virtual)...")
    wt_sol = biochem_net.integrate(C0, t_max)
    if not wt_sol.success:
        return {'error': f'WT failed: {wt_sol.message}'}
    
    wt_fluxes = compute_biomass_flux(biochem_net, wt_sol, biomass_rxn_idx)
    wt_growth = wt_fluxes['mean_flux']
    threshold = threshold_fraction * wt_growth
    
    if verbose:
        print(f"  WT growth rate: {wt_growth:.3e} mM/s")
        print(f"  Essentiality threshold: {threshold:.3e} mM/s")
        print(f"  WT final metabolite range: "
              f"[{wt_sol.y[:,-1].min():.2e}, {wt_sol.y[:,-1].max():.2f}] mM")
    
    # Essentiality per gene
    def is_essential_label(label: str) -> bool:
        return label in ('E', 'Q')
    
    results = []
    for gene, label in labels.items():
        rxn_idxs = gene_to_rxns.get(gene, [])
        if not rxn_idxs:
            continue
        
        ko_net = biochem_net.knockout(rxn_idxs)
        ko_sol = ko_net.integrate(C0, t_max)
        
        if not ko_sol.success:
            # Solver failure is a strong essentiality signal
            ko_growth = 0.0
        else:
            ko_fluxes = compute_biomass_flux(ko_net, ko_sol, biomass_rxn_idx)
            ko_growth = ko_fluxes['mean_flux']
        
        predicted_essential = ko_growth < threshold
        true_essential = is_essential_label(label)
        
        results.append({
            'gene': gene,
            'true_label': label,
            'true_essential': true_essential,
            'ko_growth_rate': ko_growth,
            'ko_growth_fraction': ko_growth / max(wt_growth, 1e-12),
            'predicted_essential': predicted_essential,
            'correct': predicted_essential == true_essential,
            'solver_success': ko_sol.success,
        })
    
    return {
        'wt_growth_rate': wt_growth,
        'threshold': threshold,
        'threshold_fraction': threshold_fraction,
        'results': results,
        'wt_sol': wt_sol,
    }
