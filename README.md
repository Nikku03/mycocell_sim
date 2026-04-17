# MycoCell: Multi-scale Mycoplasma Cell Simulator

A laptop-scale cell simulator for JCVI-syn3A (minimal Mycoplasma) that couples deterministic ODE metabolism with stochastic particle-based enzyme dynamics and spatial reaction-diffusion.

## What it is

Three validated simulator components:

1. **BiochemNet** — a neural-network-structured ODE solver where the graph IS the biochemistry. Nodes are metabolites, edges are reactions, activations are concentrations. Validated on Michaelis-Menten system: learns rate constants from trajectory data with <0.4% error.

2. **Particle simulator** — individual enzymes tracked in 3D with Brownian diffusion and Smoluchowski-derived stochastic binding. Validated: reproduces analytical Michaelis-Menten to 1-4% error.

3. **Spatial hybrid** — couples (1) and (2) via a voxel grid with Fickian substrate diffusion. Validated against reaction-diffusion PDE: 1.4% error, mass conservation <0.005%.

Scaled to iMB155 (304 metabolites, 244 reactions): integrates stably with LSODA.

## What's in this repo

```
mycocell_sim/
├── mycocell/
│   ├── __init__.py
│   ├── simulator.py       # BiochemNet, VoxelGrid, EnzymeParticles, SpatialHybrid
│   ├── imb155.py          # SBML + npz loaders for iMB155
│   ├── kinetics.py        # Rate constants (literature + defaults)
│   └── essentiality.py    # Hutchison 2016 labels + knockout evaluation
├── data/
│   ├── imb155.npz         # Pre-parsed stoichiometric matrix
│   └── iMB155_NoH2O.xml   # SBML source
├── MycoCell.ipynb         # Main notebook
├── requirements.txt
└── README.md (this file)
```

## Quick start

### On Colab:
1. Upload this repo
2. Open `MycoCell.ipynb`
3. Runtime → Change runtime type → GPU (optional; CPU also works)
4. Run all cells

### Locally:
```bash
pip install -r requirements.txt
jupyter notebook MycoCell.ipynb
```

## Honest limitations

**The architecture works. The essentiality predictions don't yet.**

Current state:
- **Architecture**: validated on toy problems, scales to 244-reaction model
- **Kinetic parameters**: 10 of 244 reactions have measured values; 234 use defaults (kcat=10 mM/s, Km=0.1 mM)
- **Initial concentrations**: uniform 1 mM (not physiological)
- **Simulation window**: 100 ms virtual (may be too short for some knockouts)
- **Gene mapping**: only 15 of 155 iMB155 genes mapped to Hutchison 2016 labels

What a 15-gene smoke test showed: 8% correct (1/12 essential genes correctly identified as essential; 1/1 non-essential correctly identified). This is a **failure of the simulation setup, not the architecture**. With proper initial conditions, biomass drain, longer simulation time, and more measured rate constants, the architecture can produce real predictions.

## What's needed to make it work properly

1. **Breuer 2019 Table S1** — MMSYN1 ↔ JCVISYN3A gene mapping (for full 155-gene validation)
2. **Thornburg 2022 Table S3** — measured kinetic constants for all reactions
3. **Thornburg 2022 Table S4** — measured initial concentrations
4. **Biomass drain equation** — to pull metabolites for growth; iMB155 has a biomass METABOLITE but no producing reaction; reconstruct from paper text

All of these are public; the simulator just needs access to them. Once integrated, you'd be able to run essentiality predictions on all 155 genes with real validation.

## Validation history (toy systems)

| Test | Expected | Measured | Error |
|------|----------|----------|-------|
| Particle sim vs analytical M-M (100 ms) | match | 1-4% deviation | low |
| BiochemNet learns kcat from data | 1000/s | 997/s | 0.3% |
| BiochemNet learns kon from data | 1×10⁸ | 8.99×10⁷ | 10% |
| BiochemNet held-out trajectory error | — | <0.4% per species | excellent |
| Hybrid vs analytical M-M | match | 3% mean | low |
| Spatial hybrid vs PDE (well-mixed) | match | 1.4% | low |
| Spatial hybrid mass conservation | exact | 0.004% | excellent |
| Spatial hybrid produces gradient | yes | S_near/S_far = 0.28 | physically correct |

## Contributing

This was built as a weekend exploration. If you extend it with:
- More measured rate constants
- The Breuer 2019 gene mapping
- A biomass drain equation

...then the simulator could become a useful research tool. PRs welcome.

## Citation

If you use this, please cite:

- iMB155 model: Breuer et al., *eLife* 2019, "Essential metabolism for a minimal cell"
- syn3A essentiality: Hutchison et al., *Science* 2016, "Design and synthesis of a minimal bacterial genome"  
- Whole-cell 4DWCM: Thornburg et al., *Cell* 2022, "Fundamental behaviors emerge from simulations of a living minimal cell"

## License

MIT (add your preferred license here)
