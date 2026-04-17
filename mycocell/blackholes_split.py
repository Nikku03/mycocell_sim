"""
mycocell.blackholes_split
=========================

Splits the single R_BIOMASS reaction into independent sub-reactions per
functional category (amino acids, DNA precursors, RNA precursors, lipids,
cofactors, ions). Each sub-reaction is a separate "black hole" that eats
only its own category of precursors.

This fixes the fundamental problem with the original R_BIOMASS: every
precursor's depletion affected the single multiplicative rate. With
independent sub-reactions, each black hole responds only to ITS own
precursors, enabling genuine discrimination between knockouts.

Usage:
  >>> from mycocell.blackholes_split import split_biomass_reaction
  >>> new_model = split_biomass_reaction(model, verbose=True)
  >>> # new_model has extra reactions and metabolites
  >>> # original R_BIOMASS is still there but gets zeroed out in kinetics
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from .blackholes import categorize_metabolite, CATEGORY_PATTERNS


def split_biomass_reaction(model: Dict, verbose: bool = True) -> Dict:
    """
    Return a new model with R_BIOMASS split into independent sub-reactions
    per category.
    
    For each non-empty category among the original biomass precursors, adds:
      - A new metabolite: M_biomass_{category}_c
      - A sub-biomass reaction: R_BIOMASS_{category} consuming category
        precursors and producing M_biomass_{category}_c
      - An exchange reaction: R_EX_biomass_{category}_c that drains it
    
    The original R_BIOMASS and R_EX_biomass_c are kept in the matrix but
    need to be zeroed in kinetics (configure_split_biomass_kinetics does this).
    
    Returns a new model dict with additional fields:
      - 'sub_biomass_rxn_indices': {category: reaction_index}
      - 'sub_biomass_exchange_indices': {category: reaction_index}
      - 'sub_biomass_met_indices': {category: metabolite_index}
      - 'category_precursor_indices': {category: np.array of substrate indices}
      - 'original_biomass_rxn_idx': preserved for reference
    """
    new_model = deepcopy(model)
    
    # Get the original biomass reaction
    orig_bm_idx = model['biomass_rxn_idx']
    if orig_bm_idx is None:
        raise ValueError("Model has no biomass reaction to split")
    
    bm_col = model['S'][:, orig_bm_idx]
    met_ids = list(model['met_ids'])
    
    # Group precursors by category
    precursors_by_category: Dict[str, List[Tuple[int, float]]] = {}
    for i, met_id in enumerate(met_ids):
        stoich = bm_col[i]
        if stoich >= 0:
            continue  # not a precursor
        cat = categorize_metabolite(met_id)
        precursors_by_category.setdefault(cat, []).append((i, -stoich))
    
    if verbose:
        print(f"  Original R_BIOMASS had {int((bm_col < 0).sum())} precursors")
        print(f"  Categories found:")
        for cat, precs in precursors_by_category.items():
            total_stoich = sum(s for _, s in precs)
            print(f"    {cat}: {len(precs)} mets, total stoich {total_stoich:.4f}")
    
    # Build new S matrix with added columns and rows
    n_mets_orig = len(met_ids)
    n_rxns_orig = model['S'].shape[1]
    
    # Plan new rows: one M_biomass_{cat}_c per category
    new_met_ids = []
    new_cat_for_met = []
    for cat in precursors_by_category.keys():
        new_met_ids.append(f'M_biomass_{cat}_c')
        new_cat_for_met.append(cat)
    n_new_mets = len(new_met_ids)
    
    # Plan new columns: one sub-biomass reaction + one exchange per category
    new_rxn_ids = []
    new_cat_for_rxn = []  # parallel list: which category each new rxn is for
    new_rxn_kind = []     # 'sub_biomass' or 'exchange'
    for cat in precursors_by_category.keys():
        new_rxn_ids.append(f'R_BIOMASS_{cat}')
        new_cat_for_rxn.append(cat)
        new_rxn_kind.append('sub_biomass')
    for cat in precursors_by_category.keys():
        new_rxn_ids.append(f'R_EX_biomass_{cat}_c')
        new_cat_for_rxn.append(cat)
        new_rxn_kind.append('exchange')
    n_new_rxns = len(new_rxn_ids)
    
    # Build extended S matrix
    n_mets_new = n_mets_orig + n_new_mets
    n_rxns_new = n_rxns_orig + n_new_rxns
    S_new = np.zeros((n_mets_new, n_rxns_new))
    S_new[:n_mets_orig, :n_rxns_orig] = model['S']
    
    # Indices for new metabolites in the extended matrix
    met_idx_for_cat = {}
    for k, (mid, cat) in enumerate(zip(new_met_ids, new_cat_for_met)):
        met_idx_for_cat[cat] = n_mets_orig + k
    
    # Fill in new columns
    sub_biomass_rxn_indices = {}
    sub_biomass_exchange_indices = {}
    
    for k, (rid, cat, kind) in enumerate(zip(new_rxn_ids, new_cat_for_rxn, new_rxn_kind)):
        col_idx = n_rxns_orig + k
        bm_met_idx = met_idx_for_cat[cat]
        
        if kind == 'sub_biomass':
            # Reactants: all precursors in this category
            for prec_met_idx, stoich in precursors_by_category[cat]:
                S_new[prec_met_idx, col_idx] = -stoich  # consumed
            # Product: one unit of the category's biomass
            S_new[bm_met_idx, col_idx] = 1.0
            sub_biomass_rxn_indices[cat] = col_idx
        elif kind == 'exchange':
            # Consumes the category biomass, produces nothing
            S_new[bm_met_idx, col_idx] = -1.0
            sub_biomass_exchange_indices[cat] = col_idx
    
    # Precursor indices per category (for kinetics config)
    category_precursor_indices = {
        cat: np.array([i for i, _ in precursors_by_category[cat]])
        for cat in precursors_by_category.keys()
    }
    category_precursor_stoichs = {
        cat: np.array([s for _, s in precursors_by_category[cat]])
        for cat in precursors_by_category.keys()
    }
    
    # Update the new model dict
    new_model['S'] = S_new
    new_model['met_ids'] = met_ids + new_met_ids
    new_model['rxn_ids'] = list(model['rxn_ids']) + new_rxn_ids
    
    # Extend reversibility and bounds arrays
    new_reversible = np.zeros(n_rxns_new, dtype=bool)
    new_reversible[:n_rxns_orig] = model['reversible']
    # All new sub-biomasses are irreversible
    new_model['reversible'] = new_reversible
    
    new_lb = np.full(n_rxns_new, 0.0)
    new_lb[:n_rxns_orig] = model['lb']
    new_ub = np.full(n_rxns_new, 1000.0)
    new_ub[:n_rxns_orig] = model['ub']
    new_model['lb'] = new_lb
    new_model['ub'] = new_ub
    
    # Record the new structure for downstream use
    new_model['sub_biomass_rxn_indices'] = sub_biomass_rxn_indices
    new_model['sub_biomass_exchange_indices'] = sub_biomass_exchange_indices
    new_model['sub_biomass_met_indices'] = met_idx_for_cat
    new_model['category_precursor_indices'] = category_precursor_indices
    new_model['category_precursor_stoichs'] = category_precursor_stoichs
    new_model['original_biomass_rxn_idx'] = orig_bm_idx
    new_model['original_biomass_exchange_idx'] = model.get('biomass_exchange_idx')
    
    if verbose:
        print(f"\n  After split:")
        print(f"    Metabolites: {n_mets_orig} → {n_mets_new} (+{n_new_mets})")
        print(f"    Reactions:   {n_rxns_orig} → {n_rxns_new} (+{n_new_rxns})")
        print(f"    Sub-biomass reactions: {len(sub_biomass_rxn_indices)}")
        for cat, idx in sub_biomass_rxn_indices.items():
            n_prec = len(category_precursor_indices[cat])
            print(f"      R_BIOMASS_{cat}: idx {idx}, {n_prec} precursors")
    
    return new_model


def configure_split_biomass_kinetics(
    split_model: Dict,
    vmax_f: np.ndarray,
    vmax_r: np.ndarray,
    km_per_rxn: List[Dict[str, float]],
    biomass_kcat: float = 1.0,
    biomass_km: float = 0.01,
    exchange_rate: float = 100.0,
    zero_original: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """
    Extend vmax and km arrays to cover the new sub-biomass reactions.
    
    Args:
        split_model: output of split_biomass_reaction()
        vmax_f, vmax_r, km_per_rxn: kinetics arrays sized for ORIGINAL model
            (they'll be extended to the new size)
        biomass_kcat: max rate for each sub-biomass reaction (1/s)
        biomass_km: Km for each precursor (mM). Use a small value so sub-biomass
            runs at near-max rate unless a precursor is depleted.
        exchange_rate: first-order drain rate for sub-biomass_c pools
        zero_original: if True, set the original R_BIOMASS and R_EX_biomass_c
            vmax to 0 so only sub-biomasses produce flux
    """
    n_rxns_new = split_model['S'].shape[1]
    n_rxns_orig = len(vmax_f)
    
    # Extend arrays
    vmax_f_new = np.concatenate([vmax_f, np.zeros(n_rxns_new - n_rxns_orig)])
    vmax_r_new = np.concatenate([vmax_r, np.zeros(n_rxns_new - n_rxns_orig)])
    km_per_rxn_new = list(km_per_rxn) + [{} for _ in range(n_rxns_new - n_rxns_orig)]
    
    # Zero original biomass if requested
    if zero_original:
        orig_idx = split_model['original_biomass_rxn_idx']
        vmax_f_new[orig_idx] = 0.0
        vmax_r_new[orig_idx] = 0.0
        if split_model.get('original_biomass_exchange_idx') is not None:
            ex_idx = split_model['original_biomass_exchange_idx']
            vmax_f_new[ex_idx] = 0.0
            vmax_r_new[ex_idx] = 0.0
    
    met_ids = split_model['met_ids']
    
    # Configure sub-biomass reactions
    for cat, rxn_idx in split_model['sub_biomass_rxn_indices'].items():
        precursor_indices = split_model['category_precursor_indices'][cat]
        
        # Kinetics: MM with small Km so reaction runs at near-vmax unless
        # a precursor drops below Km
        vmax_f_new[rxn_idx] = biomass_kcat
        vmax_r_new[rxn_idx] = 0.0
        
        km_dict = {}
        for i in precursor_indices:
            met_id = met_ids[i]
            normalized = _normalize_met(met_id)
            km_dict[normalized] = biomass_km
        km_per_rxn_new[rxn_idx] = km_dict
    
    # Configure exchange reactions
    exchange_km = 10.0  # mM — large enough that behavior is first-order
    for cat, ex_idx in split_model['sub_biomass_exchange_indices'].items():
        # vmax such that effective first-order rate is exchange_rate
        vmax_f_new[ex_idx] = exchange_rate * exchange_km
        vmax_r_new[ex_idx] = 0.0
        bm_met_idx = split_model['sub_biomass_met_indices'][cat]
        bm_met_id = met_ids[bm_met_idx]
        km_per_rxn_new[ex_idx] = {_normalize_met(bm_met_id): exchange_km}
    
    if verbose:
        print(f"  Configured kinetics for {len(split_model['sub_biomass_rxn_indices'])} "
              f"sub-biomass reactions:")
        for cat, rxn_idx in split_model['sub_biomass_rxn_indices'].items():
            n_prec = len(split_model['category_precursor_indices'][cat])
            print(f"    R_BIOMASS_{cat}: kcat={biomass_kcat}, Km={biomass_km}, "
                  f"{n_prec} precursors")
        if zero_original:
            print(f"  Zeroed original R_BIOMASS and R_EX_biomass_c")
    
    return vmax_f_new, vmax_r_new, km_per_rxn_new


def _normalize_met(mid: str) -> str:
    """Strip M_ prefix and compartment suffix."""
    s = mid
    if s.startswith('M_'):
        s = s[2:]
    if s.endswith('_c') or s.endswith('_e') or s.endswith('_p'):
        s = s[:-2]
    return s


def sub_biomass_fluxes(biochem_net, sol, split_model: Dict) -> Dict[str, np.ndarray]:
    """
    Extract flux trajectories for each sub-biomass reaction.
    
    Returns: {category: flux trajectory (n_timepoints,)}
    """
    n_t = sol.t.size
    result = {}
    
    for cat, rxn_idx in split_model['sub_biomass_rxn_indices'].items():
        fluxes = np.zeros(n_t)
        for t_idx in range(n_t):
            rates = biochem_net.compute_rates(sol.y[:, t_idx])
            fluxes[t_idx] = rates[rxn_idx]
        result[cat] = fluxes
    
    return result


def sub_biomass_summary(sub_fluxes: Dict[str, np.ndarray], sol,
                        last_fraction: float = 0.2) -> Dict[str, Dict]:
    """
    Summary statistics per sub-biomass: mean rate over last fraction of sim,
    and min/max over the whole trajectory.
    """
    n = sol.t.size
    start = int(n * (1 - last_fraction))
    
    result = {}
    for cat, flux in sub_fluxes.items():
        result[cat] = {
            'mean_late': float(flux[start:].mean()),
            'mean_overall': float(flux.mean()),
            'final': float(flux[-1]),
            'max': float(flux.max()),
            'min': float(flux.min()),
        }
    return result
