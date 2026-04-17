"""
mycocell.thornburg
==================

Adapters for the Thornburg 2022 Luthey-Schulten Lab data files.

Replaces our 16 hardcoded literature entries + 233 defaults with ~200+
measured values from `kinetic_params.xlsx`, and replaces uniform 1 mM
initial conditions with measured values from `initial_concentrations.xlsx`.

Data source:
  https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation/
  tree/main/input_data/

Usage:
  >>> from mycocell.thornburg import load_kinetics, load_initial_concentrations
  >>> kin = load_kinetics('data/thornburg_2022/kinetic_params.xlsx')
  >>> ic = load_initial_concentrations('data/thornburg_2022/initial_concentrations.xlsx')
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

# Which sheets in kinetic_params.xlsx correspond to metabolism
# (as opposed to gene expression, ribosome assembly, etc.)
METABOLISM_SHEETS = [
    'Central',
    'Nucleotide', 
    'Lipid',
    'Cofactor',
    'Transport',
    'Non-Random-Binding Reactions',
]

# Parameter type strings we recognize
PARAM_KCAT_FWD = 'Substrate Catalytic Rate Constant'
PARAM_KCAT_REV = 'Product Catalytic Rate Constant'
PARAM_KM = 'Michaelis Menten Constant'
PARAM_ENZ_COUNT = 'Eff Enzyme Count'

# Sanity-check ranges
KCAT_MIN = 1e-3   # 1/s
KCAT_MAX = 1e6    # 1/s
KM_MIN = 1e-5     # mM
KM_MAX = 1e3      # mM
CONC_MIN = 1e-6   # mM
CONC_MAX = 1e3    # mM


# ---------------------------------------------------------------
# Kinetic parameters parser
# ---------------------------------------------------------------

def load_kinetics(xlsx_path: str,
                  sheets: Optional[List[str]] = None,
                  verbose: bool = True) -> Dict:
    """
    Parse kinetic_params.xlsx into a nested dict keyed by reaction name.
    
    Returns:
      {reaction_name: {
          'kcat_f': float (1/s),
          'kcat_r': float or None (1/s),
          'km': {met_id: float (mM)},
          'enzyme': str or None,
          'subsystem': str,
      }}
    
    Flags logged to stderr:
      - reactions with kcat but no Km (suspicious)
      - reactions with Km but no kcat (weird)
      - values outside sanity ranges
    """
    if sheets is None:
        sheets = METABOLISM_SHEETS
    
    xl = pd.ExcelFile(xlsx_path)
    available = [s for s in sheets if s in xl.sheet_names]
    missing = [s for s in sheets if s not in xl.sheet_names]
    if verbose and missing:
        print(f"  Skipping missing sheets: {missing}")
    
    all_rxns: Dict[str, Dict] = {}
    flags = {'kcat_only': [], 'km_only': [], 'out_of_range': []}
    
    for sheet_name in available:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        if verbose:
            print(f"  Sheet '{sheet_name}': {len(df)} rows")
        
        # Expected columns; bail if structure differs
        required = {'Reaction Name', 'Parameter Type', 'Related Species', 'Value'}
        if not required.issubset(df.columns):
            if verbose:
                print(f"    Skipping: missing required columns. Got: {list(df.columns)}")
            continue
        
        # Group by reaction name
        for rxn_name, group in df.groupby('Reaction Name'):
            if pd.isna(rxn_name):
                continue
            rxn_name = str(rxn_name).strip()
            if rxn_name not in all_rxns:
                all_rxns[rxn_name] = {
                    'kcat_f': None,
                    'kcat_r': None,
                    'km': {},
                    'enzyme': None,
                    'subsystem': sheet_name,
                }
            entry = all_rxns[rxn_name]
            
            for _, row in group.iterrows():
                ptype = str(row['Parameter Type']).strip()
                value = row['Value']
                species = row.get('Related Species', None)
                
                if ptype == PARAM_KCAT_FWD:
                    v = _safe_float(value)
                    if v is not None:
                        if _in_range(v, KCAT_MIN, KCAT_MAX):
                            entry['kcat_f'] = v
                        else:
                            flags['out_of_range'].append(
                                (rxn_name, 'kcat_f', v))
                
                elif ptype == PARAM_KCAT_REV:
                    v = _safe_float(value)
                    if v is not None:
                        if _in_range(v, KCAT_MIN, KCAT_MAX):
                            entry['kcat_r'] = v
                        else:
                            flags['out_of_range'].append(
                                (rxn_name, 'kcat_r', v))
                
                elif ptype == PARAM_KM:
                    v = _safe_float(value)
                    if v is not None and species and not pd.isna(species):
                        species_str = str(species).strip()
                        if _in_range(v, KM_MIN, KM_MAX):
                            entry['km'][species_str] = v
                        else:
                            flags['out_of_range'].append(
                                (rxn_name, f'km[{species_str}]', v))
                
                elif ptype == PARAM_ENZ_COUNT:
                    # Value is typically a protein ID string, not a number
                    if not pd.isna(value):
                        entry['enzyme'] = str(value).strip()
    
    # Post-process: flag reactions with kcat but no Km, and vice versa
    for rxn_name, entry in all_rxns.items():
        has_kcat = entry['kcat_f'] is not None or entry['kcat_r'] is not None
        has_km = len(entry['km']) > 0
        if has_kcat and not has_km:
            flags['kcat_only'].append(rxn_name)
        if has_km and not has_kcat:
            flags['km_only'].append(rxn_name)
    
    if verbose:
        print(f"\n  Parsed {len(all_rxns)} reactions across {len(available)} sheets")
        has_both = sum(1 for e in all_rxns.values() 
                       if (e['kcat_f'] or e['kcat_r']) and e['km'])
        print(f"  With both kcat and Km: {has_both}")
        print(f"  Flags:")
        print(f"    kcat but no Km:    {len(flags['kcat_only'])}"
              + (f" e.g. {flags['kcat_only'][:3]}" if flags['kcat_only'] else ''))
        print(f"    Km but no kcat:    {len(flags['km_only'])}"
              + (f" e.g. {flags['km_only'][:3]}" if flags['km_only'] else ''))
        print(f"    Out-of-range:      {len(flags['out_of_range'])}"
              + (f" e.g. {flags['out_of_range'][:3]}" if flags['out_of_range'] else ''))
    
    return {'reactions': all_rxns, 'flags': flags}


def _safe_float(value) -> Optional[float]:
    """Return float if value is numeric, else None. Handles NaN, strings, etc."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Try to parse string as number
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _in_range(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


# ---------------------------------------------------------------
# Initial concentrations parser
# ---------------------------------------------------------------

def load_initial_concentrations(xlsx_path: str,
                                 sheet: str = 'Intracellular Metabolites',
                                 verbose: bool = True) -> Dict[str, float]:
    """
    Parse initial_concentrations.xlsx → dict of {met_id: conc_mM}.
    
    met_id is normalized to match SBML format: prepends 'M_' if not present.
    Example: 'atp_c' → 'M_atp_c'
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if verbose:
        print(f"  Sheet '{sheet}': {len(df)} rows, columns: {list(df.columns)}")
    
    # Expected columns: 'Met ID', 'Init Conc (mM)'
    id_col = None
    conc_col = None
    for c in df.columns:
        c_low = c.lower().strip()
        if 'met id' in c_low or c_low == 'id':
            id_col = c
        if 'init' in c_low and 'conc' in c_low:
            conc_col = c
    
    if id_col is None or conc_col is None:
        raise ValueError(
            f"Can't find ID or concentration columns. Got: {list(df.columns)}")
    
    if verbose:
        print(f"  Using ID column: '{id_col}', conc column: '{conc_col}'")
    
    result = {}
    flags = {'missing': [], 'out_of_range': []}
    for _, row in df.iterrows():
        met_raw = row[id_col]
        conc = row[conc_col]
        
        if pd.isna(met_raw) or pd.isna(conc):
            continue
        
        met_id = str(met_raw).strip()
        # Normalize to SBML format
        if not met_id.startswith('M_'):
            met_id = 'M_' + met_id
        
        v = _safe_float(conc)
        if v is None:
            flags['missing'].append(met_id)
            continue
        
        if not _in_range(v, CONC_MIN, CONC_MAX):
            flags['out_of_range'].append((met_id, v))
            continue
        
        result[met_id] = v
    
    if verbose:
        print(f"\n  Loaded {len(result)} initial concentrations")
        if result:
            concs = sorted(result.values())
            print(f"  Range: {concs[0]:.2e} to {concs[-1]:.2f} mM "
                  f"(median: {concs[len(concs)//2]:.2f})")
        if flags['missing']:
            print(f"  Skipped {len(flags['missing'])} rows with missing values")
        if flags['out_of_range']:
            print(f"  Skipped {len(flags['out_of_range'])} out-of-range values "
                  f"e.g. {flags['out_of_range'][:3]}")
    
    return result


# ---------------------------------------------------------------
# Assembly: combine Thornburg kinetics with iMB155 reaction list
# ---------------------------------------------------------------

def build_rate_arrays_thornburg(rxn_ids: List[str],
                                 thornburg_kin: Dict,
                                 default_vmax: float = 10.0,
                                 default_km: float = 0.1,
                                 verbose: bool = True) -> Dict:
    """
    Build vmax_f, vmax_r, km_per_rxn arrays using Thornburg measured values
    where available, defaults elsewhere.
    
    Note: vmax = kcat * [enzyme]. Thornburg gives us kcat but the full
    equation requires enzyme count. For now we treat vmax ≈ kcat (i.e.
    assume [enzyme] effectively 1 mM). In a more rigorous setup we'd
    multiply by the actual enzyme concentration from the Comparative
    Proteomics sheet.
    """
    n = len(rxn_ids)
    vmax_f = np.full(n, default_vmax)
    vmax_r = np.full(n, 1.0)
    km_per_rxn: List[Dict[str, float]] = [{} for _ in range(n)]
    is_measured = np.zeros(n, dtype=bool)
    
    rxns = thornburg_kin['reactions']
    
    for i, rid in enumerate(rxn_ids):
        # Strip R_ prefix and potential _L/_D suffix for matching
        stripped = rid[2:] if rid.startswith('R_') else rid
        candidates = [stripped]
        if stripped.endswith('_L') or stripped.endswith('_D'):
            candidates.append(stripped[:-2])
        
        matched = None
        for cand in candidates:
            if cand in rxns:
                matched = rxns[cand]
                break
        
        if matched is None:
            continue
        
        if matched['kcat_f'] is not None:
            vmax_f[i] = matched['kcat_f']
        if matched['kcat_r'] is not None:
            vmax_r[i] = matched['kcat_r']
        
        if matched['km']:
            # Normalize met IDs from 'M_g6p_c' to 'g6p' for km_per_rxn dict
            # (our BiochemNet strips M_ and compartment suffix)
            for met_id, km_val in matched['km'].items():
                normalized = _normalize_met(met_id)
                km_per_rxn[i][normalized] = km_val
        
        is_measured[i] = True
    
    if verbose:
        print(f"\n  Thornburg kinetics applied to {is_measured.sum()}/{n} "
              f"reactions in iMB155")
        unmapped = [
            (rid, rxns.keys()) for i, rid in enumerate(rxn_ids) 
            if not is_measured[i]
        ]
        if unmapped:
            print(f"  {len(unmapped)} reactions still using defaults")
    
    return {
        'vmax_f': vmax_f,
        'vmax_r': vmax_r,
        'km_per_rxn': km_per_rxn,
        'is_measured': is_measured,
        'default_km': default_km,
        'n_measured': int(is_measured.sum()),
        'n_default': int((~is_measured).sum()),
    }


def _normalize_met(mid: str) -> str:
    """Strip M_ prefix and compartment suffix."""
    s = mid
    if s.startswith('M_'):
        s = s[2:]
    if s.endswith('_c') or s.endswith('_e') or s.endswith('_p'):
        s = s[:-2]
    return s


def build_C0_from_thornburg(met_ids: List[str],
                             initial_concs: Dict[str, float],
                             default_conc: float = 1.0,
                             verbose: bool = True) -> np.ndarray:
    """
    Build initial concentration vector C0 for BiochemNet.
    
    Uses measured Thornburg concentrations where available, default_conc
    (default 1 mM) elsewhere.
    """
    n = len(met_ids)
    C0 = np.full(n, default_conc)
    matched = 0
    
    for i, mid in enumerate(met_ids):
        if mid in initial_concs:
            C0[i] = initial_concs[mid]
            matched += 1
    
    if verbose:
        print(f"\n  Initial concentrations: {matched}/{n} metabolites matched")
        print(f"  Others use default {default_conc} mM")
        print(f"  C0 range: {C0.min():.3e} to {C0.max():.2f} mM")
    
    return C0
