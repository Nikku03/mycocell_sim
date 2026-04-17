"""
mycocell.blackholes
===================

"Black hole" decomposition of the biomass reaction.

The biomass reaction lumps ~53 metabolites into one abstract sink. We don't
know the internals (translation, replication, lipid assembly) but we CAN
observe what flows in and out of each functional sub-process.

This module:
  1. Categorizes biomass precursors into functional black holes
  2. Extracts per-reaction flux from simulation trajectories
  3. Provides diagnostic plots showing flows through each black hole

Principle: we don't model what's inside. We just watch the input/output.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------
# Categorization rules for biomass precursors
# ---------------------------------------------------------------

# Each tuple: (category, list of metabolite ID patterns that match)
# Order matters — first match wins. So put more specific patterns first.
CATEGORY_PATTERNS = [
    # Amino acids (L-form standard; also glycine which has no L/D)
    ('amino_acids', [
        'M_ala__L_c', 'M_arg__L_c', 'M_asn__L_c', 'M_asp__L_c',
        'M_cys__L_c', 'M_gln__L_c', 'M_glu__L_c', 'M_gly_c',
        'M_his__L_c', 'M_ile__L_c', 'M_leu__L_c', 'M_lys__L_c',
        'M_met__L_c', 'M_phe__L_c', 'M_pro__L_c', 'M_ser__L_c',
        'M_thr__L_c', 'M_trp__L_c', 'M_tyr__L_c', 'M_val__L_c',
    ]),
    # Deoxyribonucleotides (DNA precursors)
    ('dna_precursors', [
        'M_datp_c', 'M_dctp_c', 'M_dgtp_c', 'M_dttp_c',
    ]),
    # Ribonucleotides (RNA precursors)
    ('rna_precursors', [
        'M_atp_c', 'M_ctp_c', 'M_gtp_c', 'M_utp_c',
    ]),
    # Bulk polymers already (DNA/RNA as species)
    ('polymers', [
        'M_DNA_c', 'M_RNA_c',
    ]),
    # Cofactors
    ('cofactors', [
        'M_nad_c', 'M_nadh_c', 'M_nadp_c', 'M_nadph_c',
        'M_fad_c', 'M_fadh2_c', 'M_coa_c', 'M_accoa_c',
        'M_ACP_c', 'M_dUTPase_c',
    ]),
    # Lipids and membrane components
    ('lipids', [
        'M_pc_c', 'M_sm_c', 'M_12dgr_c', 'M_chsterol_c',
        'M_clpn_c', 'M_fa_c', 'M_galfur12dgr_c', 'M_pa_c',
        'M_ps_c', 'M_pe_c', 'M_pg_c',
    ]),
    # Ions and small molecules
    ('ions', [
        'M_ca2_c', 'M_cl_c', 'M_k_c', 'M_na1_c', 'M_mg2_c',
        'M_zn2_c', 'M_fe2_c', 'M_fe3_c',
    ]),
]


def categorize_metabolite(met_id: str) -> str:
    """Return which black hole category a metabolite belongs to.
    Returns 'other' if no match."""
    for category, patterns in CATEGORY_PATTERNS:
        if met_id in patterns:
            return category
    return 'other'


def decompose_biomass(S: np.ndarray, biomass_rxn_idx: int,
                      met_ids: List[str]) -> Dict[str, Dict]:
    """
    Decompose the biomass reaction into functional black holes.
    
    For each category, return:
      {
        'metabolites': list of (met_id, stoich) in this category,
        'total_stoich': sum of stoichiometries (mass contribution to biomass),
        'met_indices': np.array of indices into met_ids,
        'stoichs': np.array of stoich coefficients (positive values),
      }
    """
    bm_col = S[:, biomass_rxn_idx]
    
    categories: Dict[str, Dict] = {}
    
    for i, met_id in enumerate(met_ids):
        stoich = bm_col[i]
        if stoich >= 0:
            continue  # not a precursor
        
        cat = categorize_metabolite(met_id)
        if cat not in categories:
            categories[cat] = {
                'metabolites': [],
                'met_indices': [],
                'stoichs': [],
            }
        categories[cat]['metabolites'].append((met_id, -stoich))
        categories[cat]['met_indices'].append(i)
        categories[cat]['stoichs'].append(-stoich)
    
    # Convert to arrays and compute totals
    for cat, data in categories.items():
        data['met_indices'] = np.array(data['met_indices'], dtype=int)
        data['stoichs'] = np.array(data['stoichs'])
        data['total_stoich'] = float(data['stoichs'].sum())
        data['n_mets'] = len(data['metabolites'])
    
    return categories


# ---------------------------------------------------------------
# Flux extraction from trajectory
# ---------------------------------------------------------------

def extract_fluxes(biochem_net, sol) -> np.ndarray:
    """
    Extract flux of every reaction at every recorded time point.
    
    Returns: (n_reactions, n_timepoints) array of fluxes in mM/s.
    """
    n_rxns = biochem_net.n_rxns
    n_t = sol.t.size
    fluxes = np.zeros((n_rxns, n_t))
    for t_idx in range(n_t):
        fluxes[:, t_idx] = biochem_net.compute_rates(sol.y[:, t_idx])
    return fluxes


def blackhole_throughput(sol, biochem_net, biomass_rxn_idx: int,
                         categories: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    For each black hole category, compute how fast it's eating each precursor
    given the current biomass flux trajectory.
    
    Ingestion rate of precursor i (in category c) = stoich_i × biomass_flux
    (by definition of the stoichiometric matrix — but we compute it
    directly from the simulation anyway, so it includes any solver effects).
    
    Returns per category:
      {
        'ingestion_rate_over_time': array of ingestion rates (summed over
                                     metabolites in category) in mM/s,
        'mean_ingestion_rate': scalar, mean over last 20% of sim,
        'bottleneck_metabolite': met_id with lowest C/C_initial at end,
        'bottleneck_fraction': C_final/C_initial of the bottleneck,
      }
    """
    fluxes = extract_fluxes(biochem_net, sol)
    biomass_flux = fluxes[biomass_rxn_idx, :]
    
    C0 = sol.y[:, 0]
    C_final = sol.y[:, -1]
    
    result = {}
    for cat, data in categories.items():
        # Total ingestion for this category = biomass_flux × sum of stoichs
        ingestion = biomass_flux * data['total_stoich']
        
        # Find bottleneck: precursor with smallest C_final / C_initial ratio
        indices = data['met_indices']
        ratios = np.where(
            C0[indices] > 1e-12,
            C_final[indices] / np.maximum(C0[indices], 1e-12),
            1.0,  # if initial was ~0, don't flag as bottleneck
        )
        bottleneck_local = np.argmin(ratios)
        bottleneck_idx_global = indices[bottleneck_local]
        bottleneck_met_id = data['metabolites'][bottleneck_local][0]
        bottleneck_ratio = float(ratios[bottleneck_local])
        
        # Mean over last 20%
        n = sol.t.size
        start = int(n * 0.8)
        mean_rate = float(ingestion[start:].mean())
        
        result[cat] = {
            'ingestion_rate_over_time': ingestion,
            'mean_ingestion_rate': mean_rate,
            'bottleneck_metabolite': bottleneck_met_id,
            'bottleneck_fraction': bottleneck_ratio,
            'bottleneck_global_idx': bottleneck_idx_global,
            'n_metabolites': data['n_mets'],
            'total_stoich': data['total_stoich'],
        }
    
    return result


# ---------------------------------------------------------------
# Diagnostic: compare WT vs knockout
# ---------------------------------------------------------------

def compare_knockout_to_wt(
    wt_throughput: Dict[str, Dict],
    ko_throughput: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    For each black hole, compute how much its ingestion dropped under knockout.
    
    Returns:
      {
        category: {
          'wt_rate': WT mean ingestion rate,
          'ko_rate': KO mean ingestion rate,
          'fraction_of_wt': ko_rate / wt_rate (0 = dead, 1 = unchanged),
          'bottleneck_shifted': did the bottleneck metabolite change?,
          'wt_bottleneck': WT bottleneck met_id,
          'ko_bottleneck': KO bottleneck met_id,
        }
      }
    """
    result = {}
    for cat in wt_throughput:
        if cat not in ko_throughput:
            continue
        wt_rate = wt_throughput[cat]['mean_ingestion_rate']
        ko_rate = ko_throughput[cat]['mean_ingestion_rate']
        fraction = ko_rate / max(abs(wt_rate), 1e-12) if wt_rate > 0 else 0.0
        
        result[cat] = {
            'wt_rate': wt_rate,
            'ko_rate': ko_rate,
            'fraction_of_wt': fraction,
            'wt_bottleneck': wt_throughput[cat]['bottleneck_metabolite'],
            'ko_bottleneck': ko_throughput[cat]['bottleneck_metabolite'],
            'bottleneck_shifted': (
                wt_throughput[cat]['bottleneck_metabolite']
                != ko_throughput[cat]['bottleneck_metabolite']),
        }
    return result


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------

def plot_blackhole_fluxes(sol, throughput: Dict[str, Dict],
                          title: str = "Black hole throughput"):
    """Plot ingestion rate over time for each category."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat, data in throughput.items():
        ax.plot(sol.t, data['ingestion_rate_over_time'],
                label=f"{cat} (n={data['n_metabolites']})",
                linewidth=2)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Ingestion rate (mM/s)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_ko_comparison(comparison: Dict[str, Dict], ko_label: str,
                        figsize: Tuple[int, int] = (10, 5)):
    """Bar chart comparing WT vs KO ingestion rates across black holes."""
    import matplotlib.pyplot as plt
    
    cats = list(comparison.keys())
    wt_rates = [comparison[c]['wt_rate'] for c in cats]
    ko_rates = [comparison[c]['ko_rate'] for c in cats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(cats))
    width = 0.35
    ax1.bar(x - width/2, wt_rates, width, label='WT', color='steelblue')
    ax1.bar(x + width/2, ko_rates, width, label=f'KO: {ko_label}',
            color='indianred')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=30, ha='right')
    ax1.set_ylabel('Ingestion rate (mM/s)')
    ax1.set_title(f'Absolute flux: WT vs {ko_label} KO')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Fraction of WT
    fracs = [comparison[c]['fraction_of_wt'] for c in cats]
    colors = ['indianred' if f < 0.5 else 'goldenrod' if f < 0.9 else 'steelblue'
              for f in fracs]
    ax2.bar(x, fracs, color=colors)
    ax2.axhline(y=1.0, linestyle='--', color='gray', alpha=0.5)
    ax2.axhline(y=0.5, linestyle=':', color='red', alpha=0.5,
                label='50% threshold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, rotation=30, ha='right')
    ax2.set_ylabel('Fraction of WT rate')
    ax2.set_title(f'Relative throughput: {ko_label} / WT')
    ax2.set_ylim(0, max(1.1, max(fracs) * 1.1))
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig
