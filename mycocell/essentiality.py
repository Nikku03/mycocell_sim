"""
mycocell.essentiality
=====================

Experimental essentiality labels (Hutchison 2016) and knockout-based
prediction using BiochemNet simulation.

Note: iMB155 uses MMSYN1_XXXX gene nomenclature (JCVI-syn1.0), while
Hutchison 2016 uses JCVISYN3A_XXXX (syn3A). The cross-genome mapping is
incomplete in the current public data; this module provides the 15-gene
subset derived from shared reaction names (glycolysis, pentose phosphate).

For the full 155-gene mapping, see Breuer 2019 supplementary table S1.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------
# Hutchison 2016 essentiality labels for JCVI-syn3A genes
# Codes: 'E' = essential, 'Q' = quasi-essential, 'N' = non-essential
# ---------------------------------------------------------------

HUTCHISON_LABELS: Dict[str, str] = {
    # Glycolysis (essential for energy)
    'JCVISYN3A_0685': 'E',  # ptsG - glucose PTS
    'JCVISYN3A_0233': 'E',  # pgi
    'JCVISYN3A_0207': 'E',  # pfkA
    'JCVISYN3A_0352': 'E',  # fba
    'JCVISYN3A_0353': 'E',  # tpiA
    'JCVISYN3A_0314': 'E',  # gapA
    'JCVISYN3A_0315': 'E',  # pgk
    'JCVISYN3A_0689': 'E',  # pgm
    'JCVISYN3A_0231': 'E',  # eno
    'JCVISYN3A_0546': 'E',  # pyk
    
    # Fermentation (non-essential)
    'JCVISYN3A_0449': 'N',  # ldh
    
    # Pentose phosphate pathway (mixed)
    'JCVISYN3A_0439': 'Q',  # zwf
    'JCVISYN3A_0441': 'E',  # gnd
    
    # Nucleotide salvage (essential)
    'JCVISYN3A_0005': 'E',  # adk
    'JCVISYN3A_0416': 'E',  # ndk
}


# ---------------------------------------------------------------
# Mapping from JCVI-syn3A gene IDs to iMB155 reaction short names
# (For the 15 genes where we have both a label and a clear single reaction)
# ---------------------------------------------------------------

GENE_TO_RXN_NAME: Dict[str, str] = {
    'JCVISYN3A_0685': 'GLCpts',
    'JCVISYN3A_0233': 'PGI',
    'JCVISYN3A_0207': 'PFK',
    'JCVISYN3A_0352': 'FBA',
    'JCVISYN3A_0353': 'TPI',
    'JCVISYN3A_0314': 'GAPD',
    'JCVISYN3A_0315': 'PGK',
    'JCVISYN3A_0689': 'PGM',
    'JCVISYN3A_0231': 'ENO',
    'JCVISYN3A_0546': 'PYK',
    'JCVISYN3A_0449': 'LDH',    # maps to R_LDH_L in iMB155
    'JCVISYN3A_0439': 'G6PDH',
    'JCVISYN3A_0441': 'GND',
    'JCVISYN3A_0005': 'ADK',
    'JCVISYN3A_0416': 'NDK',
}


def is_essential(label: str) -> bool:
    """Convert Hutchison code to binary essential flag. E or Q = essential."""
    return label in ('E', 'Q')


def assess_viability(sol, wt_final: Optional[np.ndarray] = None,
                     fail_threshold: float = 0.001,
                     explode_threshold: float = 1000.0) -> str:
    """
    Classify a simulation outcome.
    
    Returns one of:
      - 'solver_failed': ODE solver couldn't integrate
      - 'exploded': more than 5 metabolites above explode_threshold
      - 'crashed': more than 30 metabolites below fail_threshold
      - 'deviant': final state very different from WT (requires wt_final)
      - 'viable': none of the above
    """
    if sol is None or not sol.success:
        return 'solver_failed'
    
    C_final = sol.y[:, -1]
    n_depleted = int((C_final < fail_threshold).sum())
    n_exploded = int((C_final > explode_threshold).sum())
    
    if n_exploded > 5:
        return 'exploded'
    if n_depleted > 30:
        return 'crashed'
    
    if wt_final is not None:
        wt_norm = np.maximum(wt_final, 0.01)
        rel_diff = np.abs(C_final - wt_final) / wt_norm
        if np.median(rel_diff) > 0.5 or np.max(rel_diff) > 100:
            return 'deviant'
    
    return 'viable'


def evaluate_essentiality(biochem_net, C0: np.ndarray,
                           gene_to_rxn_indices: Dict[str, List[int]],
                           labels: Dict[str, str],
                           t_max: float = 0.1) -> List[Dict]:
    """
    For each gene in labels, knock out its reactions, simulate, and classify.
    
    Returns list of dicts with predicted and true essentiality per gene.
    """
    # WT baseline
    sol_wt = biochem_net.integrate(C0, t_max)
    wt_final = sol_wt.y[:, -1] if sol_wt.success else None
    wt_verdict = assess_viability(sol_wt, C0)
    
    results = []
    for gene, label in labels.items():
        rxn_idxs = gene_to_rxn_indices.get(gene, [])
        if not rxn_idxs:
            continue
        
        # Knock out this gene's reactions
        ko_net = biochem_net.knockout(rxn_idxs)
        ko_sol = ko_net.integrate(C0, t_max)
        ko_verdict = assess_viability(ko_sol, wt_final)
        
        predicted_essential = ko_verdict in (
            'solver_failed', 'exploded', 'crashed', 'deviant')
        true_essential = is_essential(label)
        
        results.append({
            'gene': gene,
            'true_label': label,
            'true_essential': true_essential,
            'ko_verdict': ko_verdict,
            'predicted_essential': predicted_essential,
            'correct': predicted_essential == true_essential,
            'n_reactions_knocked': len(rxn_idxs),
        })
    
    return {
        'wt_verdict': wt_verdict,
        'results': results,
    }
