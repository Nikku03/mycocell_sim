"""
mycocell.imb155
===============

Loaders for the iMB155 minimal Mycoplasma metabolic model.

Data files required (placed in ./data/):
  - imb155.npz          stoichiometric matrix and bounds
  - iMB155_NoH2O.xml    SBML file for gene-reaction associations
"""

from __future__ import annotations
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional


SBML_NS = {
    'sbml': 'http://www.sbml.org/sbml/level3/version1/core',
    'fbc':  'http://www.sbml.org/sbml/level3/version1/fbc/version2',
}


def load_stoichiometry(npz_path: str) -> Dict:
    """Load pre-parsed stoichiometric matrix from npz."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'S': data['S'],            # (304 mets, 244 rxns)
        'lb': data['lb'],
        'ub': data['ub'],
        'reversible': data['reversible'],
        'E': data['E'],            # elemental composition matrix
        'C_null': data['C_null'],  # conservation laws
    }


def load_sbml_annotations(sbml_path: str) -> Dict:
    """Parse SBML for metabolite IDs, reaction IDs, and gene associations."""
    tree = ET.parse(sbml_path)
    root = tree.getroot()
    model = root.find('sbml:model', SBML_NS)
    
    # Metabolite IDs in S-matrix order
    met_ids = [sp.get('id')
               for sp in model.find('sbml:listOfSpecies', SBML_NS)
                             .findall('sbml:species', SBML_NS)]
    
    # Reaction IDs in S-matrix order
    rxn_ids = [r.get('id')
               for r in model.find('sbml:listOfReactions', SBML_NS)
                             .findall('sbml:reaction', SBML_NS)]
    
    # Gene list with labels
    gene_labels = {}  # fbc:geneProduct id -> label
    gp_list = model.find('fbc:listOfGeneProducts', SBML_NS)
    if gp_list is not None:
        for gp in gp_list.findall('fbc:geneProduct', SBML_NS):
            gid = gp.get('{%s}id' % SBML_NS['fbc'])
            label = gp.get('{%s}label' % SBML_NS['fbc'], gid)
            gene_labels[gid] = label
    
    # Reaction -> genes association
    rxn_to_genes = {}
    for r in model.find('sbml:listOfReactions', SBML_NS).findall(
            'sbml:reaction', SBML_NS):
        rid = r.get('id')
        genes = []
        gpa = r.find('fbc:geneProductAssociation', SBML_NS)
        if gpa is not None:
            for gref in gpa.iter('{%s}geneProductRef' % SBML_NS['fbc']):
                genes.append(gref.get('{%s}geneProduct' % SBML_NS['fbc']))
        rxn_to_genes[rid] = genes
    
    return {
        'met_ids': met_ids,
        'rxn_ids': rxn_ids,
        'gene_labels': gene_labels,   # fbc_id -> label (MMSYN1_XXXX)
        'rxn_to_genes': rxn_to_genes,  # rxn_id -> list of fbc_ids
    }


def build_gene_to_rxn_indices(
        rxn_ids: List[str], rxn_to_genes: Dict[str, List[str]]
) -> Dict[str, List[int]]:
    """Build gene_id -> list of reaction indices in S."""
    rxn_id_to_idx = {rid: i for i, rid in enumerate(rxn_ids)}
    gene_to_rxns = {}
    for rid, genes in rxn_to_genes.items():
        idx = rxn_id_to_idx.get(rid)
        if idx is None:
            continue
        for g in genes:
            gene_to_rxns.setdefault(g, []).append(idx)
    return gene_to_rxns


def load_model(data_dir: str = 'data') -> Dict:
    """Load the full iMB155 model from stoichiometry + SBML."""
    data_dir = Path(data_dir)
    stoich = load_stoichiometry(str(data_dir / 'imb155.npz'))
    annot = load_sbml_annotations(str(data_dir / 'iMB155_NoH2O.xml'))
    
    # Sanity check
    if len(annot['met_ids']) != stoich['S'].shape[0]:
        raise ValueError(
            f"Metabolite count mismatch: SBML has {len(annot['met_ids'])}, "
            f"S matrix has {stoich['S'].shape[0]} rows")
    if len(annot['rxn_ids']) != stoich['S'].shape[1]:
        raise ValueError(
            f"Reaction count mismatch: SBML has {len(annot['rxn_ids'])}, "
            f"S matrix has {stoich['S'].shape[1]} columns")
    
    gene_to_rxns = build_gene_to_rxn_indices(
        annot['rxn_ids'], annot['rxn_to_genes'])
    
    return {
        **stoich, **annot,
        'gene_to_rxns': gene_to_rxns,
    }


def find_reaction_index(rxn_ids: List[str], short_name: str) -> Optional[int]:
    """Find reaction index by common name (handles R_ prefix, _L/_D suffix)."""
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
