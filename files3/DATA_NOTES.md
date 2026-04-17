# Data Notes: Thornburg 2022 input files

Observations from actually opening the Luthey-Schulten `Minimal_Cell_ComplexFormation` input data, recorded so we don't have to re-discover them.

## Files

All from `https://github.com/Luthey-Schulten-Lab/Minimal_Cell_ComplexFormation/tree/main/input_data/`:

| File | Size | What's in it |
|---|---|---|
| `Syn3A_updated.xml` | 364 KB | SBML model with biomass equation (replaces iMB155_NoH2O.xml) |
| `initial_concentrations.xlsx` | (small) | 140 metabolites with measured concentrations in mM |
| `kinetic_params.xlsx` | 59 KB | Rate constants across 11 subsystems |
| `complex_formation.xlsx` | — | Protein complex composition (not needed for metabolism) |
| `SSU_assembly_raw.json` | — | Ribosome assembly (not needed for metabolism) |
| `syn3A.gb` | — | GenBank genome file |

**Filename gotcha:** The README says `initial_concentration.xlsx` (singular). The actual file is `initial_concentrations.xlsx` (plural). Four wasted fetch attempts saved by noting this.

## `initial_concentrations.xlsx` structure

**Sheets:**
- `Comparative Proteomics` — protein counts for CME (not needed for ODE)
- `Simulation Medium` — what's in the growth medium
- `Intracellular Metabolites` — **this is what we want** for ODE initial conditions
- `mRNA Count` — transcripts (CME)
- `Protein Metabolites` — protein IDs used in rxns_ODE.py
- `Experimental Medium`, `Protein Topologies`, `Scaled Proteme` — other metadata

### Intracellular Metabolites sheet
140 rows × 5 columns:

| Metabolite name | Met ID | KEGG ID | InChI key | Init Conc (mM) |
|---|---|---|---|---|
| cholesterol | chsterol_c | C00187 | HVYWMOMLDIMFJA-... | 23.30 |
| 1,2-diacylglycerol | 12dgr_c | C00641 | | 6.99 |
| cardiolipin | clpn_c | C05980 | | 5.21 |
| ATP | atp_c | C00002 | ZKHQWZAMYRWXGA-... | 3.65 |
| Calcium | ca2_c | C00076 | BHPQYMZQTOCNFJ-... | 1.41 |
| 3-Phospho-D-glycerate | 3pg_c | C00197 | OSJPPGNTCRNQQC-... | 1.10 |
| Acetyl phosphate | actp_c | C00227 | LIPOUNRJVLNBCD-... | 0.63 |
| CMP | cmp_c | C00055 | IERHLVCPSMICTF-... | 0.34 |
| CDP | cdp_c | C00112 | ZWIADYZPOWUWEW-... | 0.34 |
| Acetyl-CoA | accoa_c | C00024 | ZSLZBFCDCINBPY-... | 0.25 |

**Column name to use in pandas:** `'Init Conc (mM)'` — exact spelling matters.
**Met ID format:** `chsterol_c`, `atp_c`, `3pg_c` — no `M_` prefix. Our current iMB155 uses `M_atp_c`-style IDs. Need to prepend `M_` or strip `M_` when joining.

## `kinetic_params.xlsx` structure

**Sheets:**
- `Central` (160 rows) — central metabolism (glycolysis, PPP, etc.)
- `Nucleotide` — nucleotide salvage/synthesis
- `Lipid` — lipid metabolism
- `Cofactor` — cofactor biosynthesis
- `Transport` — transport reactions
- `Non-Random-Binding Reactions` — passive transport and serial phosphorelay
- `tRNA Charging`, `Gene Expression`, `SSU Assembly`, `LSU Assembly`, `LSU Assembly w. 4 L7s` — **not metabolism, skip these for ODE**

### Central sheet structure
160 rows × 6 columns:

| Reaction Name | Subsystem | Parameter Type | Related Species | Value | Units |
|---|---|---|---|---|---|
| PGI | Central metabolism | Eff Enzyme Count | NaN | P_0445 | # |
| PGI | Central metabolism | Substrate Catalytic Rate Constant | NaN | 804.34 | 1/s |
| PGI | Central metabolism | Product Catalytic Rate Constant | NaN | 650 | 1/s |
| PGI | Central metabolism | Michaelis Menten Constant | M_g6p_c | 0.28 | mM |
| PGI | Central metabolism | Michaelis Menten Constant | M_f6p_c | 0.15 | mM |

**Parameter Types seen so far:**
- `Eff Enzyme Count` — protein id (NOT a number) indicating which enzyme catalyzes this
- `Substrate Catalytic Rate Constant` — **forward kcat** (1/s)
- `Product Catalytic Rate Constant` — **reverse kcat** (1/s)
- `Michaelis Menten Constant` — Km per metabolite (mM)
- (likely more: inhibition constants, Hill coefficients — check when parsing)

**Shape:** each Reaction Name has multiple rows, one per parameter. Need to groupby Reaction Name and pivot.

**Units:** mostly `1/s` for rates and `mM` for concentrations. Value column is strings in at least one case (enzyme count is `P_0445`), so can't blindly cast to float.

## Integration plan (adapter work)

To replace our 16 hardcoded literature entries + 233 defaults:

1. For each metabolism sheet (Central, Nucleotide, Lipid, Cofactor, Transport, Non-Random-Binding):
   - Read with pandas
   - Groupby `Reaction Name`
   - For each group, extract:
     - kcat_f from `Substrate Catalytic Rate Constant`
     - kcat_r from `Product Catalytic Rate Constant`
     - Km dict from rows where `Parameter Type == 'Michaelis Menten Constant'`, keyed by `Related Species`
   - Build a big dict: `{rxn_name: {'kcat_f': X, 'kcat_r': Y, 'km': {met: val}}}`

2. Match against iMB155 reaction IDs:
   - Our iMB155 uses `R_PGI`, `R_PFK`, etc.
   - Thornburg sheet uses `PGI`, `PFK` (no prefix)
   - Strip `R_` for matching

3. For initial concentrations:
   - Load `Intracellular Metabolites` sheet
   - Build dict `{met_id: conc}` where met_id normalized to match SBML format
   - For metabolites not in the 140-row table, fall back to 1 mM default

4. Unit sanity checks:
   - kcat should be 10 to 10^5 /s for real enzymes
   - Km should be 10^-4 to 10^2 mM
   - Flag anything outside these ranges during parsing

5. For Syn3A_updated.xml parsing:
   - Re-run our SBML parser on this new file
   - Check: how many mets, how many rxns? (expected: more than iMB155's 304/244)
   - Confirm biomass equation is present as a reaction (not just a metabolite)

## Known risks

- **Reaction name mismatches:** Thornburg's `PGI` may correspond to iMB155's `R_PGI`, but some reactions could have been renamed between papers.
- **Direction sign:** "Substrate Catalytic Rate Constant" assumes reaction is written substrate→product. We need to verify the stoichiometric direction in SBML matches.
- **Equilibrium consistency:** if kcat_f and kcat_r are both given, they implicitly define an equilibrium constant. Need to check it's physically reasonable (Haldane relation).
- **Compartment issues:** some metabolites have `_c` (cytoplasm) and `_e` (extracellular) variants. Make sure we match the right one.
