"""
mycocell.syn3a
==============

Loader for Syn3A_updated.xml — the Thornburg 2022 SBML model.

This is our "upgrade" from iMB155: 308 metabolites, 356 reactions, and
crucially, includes a biomass reaction (R_BIOMASS → M_biomass_c) and
an exchange reaction (R_EX_biomass_c consuming M_biomass_c) that
together provide a growth sink for the cell.

Parsed output matches the format our BiochemNet expects:
  {
    'S': (n_mets, n_rxns) stoichiometric matrix,
    'met_ids': list of metabolite IDs,
    'rxn_ids': list of reaction IDs,
    'reversible': (n_rxns,) bool,
    'lb', 'ub': flux bounds per reaction,
    'gene_labels': dict {gp_id: MMSYN1_XXXX},
    'rxn_to_genes': dict {rxn_id: [gp_ids]},
    'gene_to_rxns': dict {gp_id: [rxn_indices]},
    'biomass_rxn_idx': int (index of R_BIOMASS in rxn_ids),
    'biomass_exchange_idx': int (index of R_EX_biomass_c),
  }
"""

from __future__ import annotations
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


SBML_NS = {
    'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
    'fbc':  'http://www.sbml.org/sbml/level3/version1/fbc/version2',
}

# Common COBRA flux bound aliases
COBRA_BOUND_DEFAULTS = {
    'cobra_default_lb': -1000.0,
    'cobra_default_ub': 1000.0,
    'cobra_0_bound': 0.0,
    'minus_inf': -1e6,
    'plus_inf': 1e6,
}


def load_syn3a(xml_path: str, verbose: bool = True) -> Dict:
    """Parse Syn3A_updated.xml into our standard model dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    model = root.find('sbml:model', SBML_NS)
    if model is None:
        raise ValueError(f"No <model> element found in {xml_path}")
    
    # -------- parameter lookup for flux bounds --------
    # fbc-v2 stores numeric bounds as <parameter id="cobra_default_lb" value="-1000" />
    params = {}
    params_list = model.find('sbml:listOfParameters', SBML_NS)
    if params_list is not None:
        for p in params_list.findall('sbml:parameter', SBML_NS):
            pid = p.get('id')
            pval = p.get('value')
            if pval is not None:
                try:
                    params[pid] = float(pval)
                except ValueError:
                    pass
    # Merge with defaults
    for k, v in COBRA_BOUND_DEFAULTS.items():
        params.setdefault(k, v)
    
    # -------- species (metabolites) --------
    species_list = model.find('sbml:listOfSpecies', SBML_NS)
    met_ids = []
    for sp in species_list.findall('sbml:species', SBML_NS):
        met_ids.append(sp.get('id'))
    
    met_id_to_idx = {m: i for i, m in enumerate(met_ids)}
    
    # -------- reactions --------
    rxn_list = model.find('sbml:listOfReactions', SBML_NS)
    rxn_elems = rxn_list.findall('sbml:reaction', SBML_NS)
    
    n_mets = len(met_ids)
    n_rxns = len(rxn_elems)
    
    S = np.zeros((n_mets, n_rxns), dtype=np.float64)
    rxn_ids = []
    reversible = np.zeros(n_rxns, dtype=bool)
    lb = np.full(n_rxns, -1000.0)
    ub = np.full(n_rxns, 1000.0)
    rxn_to_genes: Dict[str, List[str]] = {}
    
    biomass_rxn_idx = None
    biomass_exchange_idx = None
    
    for j, r in enumerate(rxn_elems):
        rid = r.get('id')
        rxn_ids.append(rid)
        reversible[j] = r.get('reversible', 'false').lower() == 'true'
        
        # Flux bounds
        lb_ref = r.get('{%s}lowerFluxBound' % SBML_NS['fbc'])
        ub_ref = r.get('{%s}upperFluxBound' % SBML_NS['fbc'])
        if lb_ref is not None:
            lb[j] = params.get(lb_ref, -1000.0)
        if ub_ref is not None:
            ub[j] = params.get(ub_ref, 1000.0)
        
        # Reactants (negative stoichiometry)
        reacts = r.find('sbml:listOfReactants', SBML_NS)
        if reacts is not None:
            for sp_ref in reacts.findall('sbml:speciesReference', SBML_NS):
                met = sp_ref.get('species')
                stoich = float(sp_ref.get('stoichiometry', '1'))
                idx = met_id_to_idx.get(met)
                if idx is not None:
                    S[idx, j] -= stoich
        
        # Products (positive stoichiometry)
        prods = r.find('sbml:listOfProducts', SBML_NS)
        if prods is not None:
            for sp_ref in prods.findall('sbml:speciesReference', SBML_NS):
                met = sp_ref.get('species')
                stoich = float(sp_ref.get('stoichiometry', '1'))
                idx = met_id_to_idx.get(met)
                if idx is not None:
                    S[idx, j] += stoich
        
        # Gene associations
        gpa = r.find('fbc:geneProductAssociation', SBML_NS)
        genes = []
        if gpa is not None:
            for gref in gpa.iter('{%s}geneProductRef' % SBML_NS['fbc']):
                genes.append(gref.get('{%s}geneProduct' % SBML_NS['fbc']))
        rxn_to_genes[rid] = genes
        
        if rid == 'R_BIOMASS':
            biomass_rxn_idx = j
        elif rid == 'R_EX_biomass_c':
            biomass_exchange_idx = j
    
    # -------- gene products --------
    gene_labels = {}
    gp_list = model.find('fbc:listOfGeneProducts', SBML_NS)
    if gp_list is not None:
        for gp in gp_list.findall('fbc:geneProduct', SBML_NS):
            gid = gp.get('{%s}id' % SBML_NS['fbc'])
            label = gp.get('{%s}label' % SBML_NS['fbc'], gid)
            gene_labels[gid] = label
    
    # -------- gene → reaction indices --------
    gene_to_rxns: Dict[str, List[int]] = {}
    for j, rid in enumerate(rxn_ids):
        for g in rxn_to_genes.get(rid, []):
            gene_to_rxns.setdefault(g, []).append(j)
    
    # -------- sanity checks + verbose summary --------
    if verbose:
        print(f"  Parsed: {n_mets} metabolites, {n_rxns} reactions, "
              f"{len(gene_labels)} gene products")
        n_rev = int(reversible.sum())
        print(f"  Reversible: {n_rev}, Irreversible: {n_rxns - n_rev}")
        
        if biomass_rxn_idx is not None:
            bm_col = S[:, biomass_rxn_idx]
            n_consumed = int((bm_col < 0).sum())
            n_produced = int((bm_col > 0).sum())
            print(f"  Biomass reaction R_BIOMASS at index {biomass_rxn_idx}:")
            print(f"    consumes {n_consumed} metabolites, produces {n_produced}")
        else:
            print(f"  ⚠ R_BIOMASS not found")
        
        if biomass_exchange_idx is not None:
            print(f"  Biomass exchange R_EX_biomass_c at index {biomass_exchange_idx}")
        
        # Check mass balance of a simple reaction (PGI: g6p <-> f6p, should balance)
        if 'R_PGI' in rxn_ids:
            pgi_col = S[:, rxn_ids.index('R_PGI')]
            nz = np.where(pgi_col != 0)[0]
            print(f"  Sanity (R_PGI): {len(nz)} metabolites involved, "
                  f"stoich values {sorted(pgi_col[nz])}")
    
    return {
        'S': S,
        'met_ids': met_ids,
        'rxn_ids': rxn_ids,
        'reversible': reversible,
        'lb': lb,
        'ub': ub,
        'gene_labels': gene_labels,
        'rxn_to_genes': rxn_to_genes,
        'gene_to_rxns': gene_to_rxns,
        'biomass_rxn_idx': biomass_rxn_idx,
        'biomass_exchange_idx': biomass_exchange_idx,
    }


def find_reaction_index(rxn_ids: List[str], short_name: str) -> Optional[int]:
    """Find reaction index by common name. Handles R_ prefix and _L/_D suffix."""
    candidates = [
        f'R_{short_name}', f'R_{short_name}_L',
        f'R_{short_name}_D', short_name,
    ]
    for c in candidates:
        if c in rxn_ids:
            return rxn_ids.index(c)
    # Substring fallback
    for i, rid in enumerate(rxn_ids):
        if (short_name in rid and
            ('_' + short_name in rid or rid.endswith(short_name))):
            return i
    return None
