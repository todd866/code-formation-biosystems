# Code Formation at Dimensional Boundaries

**Finite-bandwidth viability optimization forces nontrivial codes when the substrate contains action-incompatible regimes.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Biology is full of codes — codons, neural spikes, receptor–ligand interfaces, morphogenetic signals, immune motifs, hormones, behavioral displays. Existing theories describe what such codes do; this paper develops a constraint-based account of why they form. The central setup is a *dimensional boundary*: an interface where the relevant state space on one side has more task-relevant degrees of freedom than the channel can transmit. Under finite bandwidth and viability pressure, retained successful protocols are partitions of substrate state with action-coherent fibers. When the substrate contains mutually action-incompatible regimes, those partitions are *forced* to allocate distinct codewords to them — codes are not merely admitted, they are required.

## Key Results

- **Structural characterization (Prop 2)**: Retained low-loss protocols at finite-bandwidth boundaries are partitions of substrate state, with fibers approximately coherent under their assigned responses; this follows from rate-distortion together with non-recoverability of distinctions discarded by projection.
- **Reachability (Props 3–4)**: An alternating Lloyd–Bayes improvement dynamics on encoder–decoder pairs decreases risk monotonically and reaches partition-shaped stationary protocols, in finitely many steps under standard tie-handling for finite/discretized substrate distributions.
- **Forced nontrivial codes (Prop 6)**: When the substrate contains `r` mutually pointwise γ-action-incompatible regimes of mass at least `p`, any protocol with regret below `γ p` must use at least `r` distinct reliable codewords; below that capacity, an explicit regret floor of `γ p` acts.
- **Conditional universality (§5, Cor 7)**: Six axioms of bounded self-maintenance — autopoietic closure, non-equilibrium maintenance, finite throughput, high-dimensional interior, action-incompatible regimes, retention — independently established across autopoiesis, non-equilibrium thermodynamics, complexity theory, life-history theory, and selection theory, jointly imply the forced-code conditions. Every living system therefore forms at least one nontrivial code at at least one of its boundaries.
- **Code–context as the operational unit (Props 7–9)**: When the substrate is entrained to a slow trajectory, anticipatory information can reside in the boundary message, in the receiver's entrained context, or in both. Stable codewords can acquire different meanings at gating events without the message itself visibly preserving phase.
- **Falsifiable prediction**: Phase-disruption experiments should selectively impair anticipation while leaving present-state classification approximately intact, dissociating the code's instantaneous identity from its predictive horizon.

## Running the Simulations

Dependencies: `numpy`, `matplotlib`. Both are imported by all scripts; nothing else is required.

```bash
pip install numpy matplotlib
```

Then from the repository root:

```bash
python3 simulations/fig1_schematic.py                    # Fig 1 schematic (forced codeword separation)
python3 simulations/generate_figures.py                  # one-bit slow-mode partition demo (Fig 2)
python3 simulations/emergence_demo.py                    # selection-driven emergence (Fig 3)
python3 simulations/first_code_complex_capacity_sweep.py # capacity sweep with codeword-usage entropy (Fig 4)
python3 simulations/first_code_complex_sweep.py          # robustness across substrate dimension N
python3 simulations/code_vs_bulk_ablation.py             # bulk-channel synchronization ablation
python3 simulations/stochastic_resonance_sweep.py        # coupled-Kuramoto stochastic-resonance demo
```

All scripts fix RNG seeds in their config blocks; rerunning from a clean checkout reproduces the reported numerical outputs to within floating-point tolerance. Outputs land in `figures/` as both `.pdf`/`.png` plots and deterministic `.txt` results files.

## Building the Manuscript

Requires a TeX distribution with `pdflatex`, `bibtex`, `natbib`, and `hyperref`.

```bash
pdflatex code_formation
bibtex code_formation
pdflatex code_formation
pdflatex code_formation
```

The repository ships a pre-built `code_formation.pdf` (48 pp).

## Paper

**Code Formation at Dimensional Boundaries: Finite-Bandwidth Coordination and Anticipation in Biology**

Todd, I. (2026). Target: *BioSystems* (in preparation).

This paper continues a sequence on information-theoretic constraints in biological substrates:

1. [Limits of Falsifiability](https://doi.org/10.1016/j.biosystems.2025.105608) — *BioSystems* 258, 105608 (2025).
2. [Timing Inaccessibility](https://doi.org/10.1016/j.biosystems.2025.105632) — *BioSystems* 258, 105632 (2025).
3. [Intelligence as High-Dimensional Coherence](https://doi.org/10.1016/j.biosystems.2026.105704) — *BioSystems* (2026).
4. [Coherence Time in Biological Oscillator Assemblies](https://doi.org/10.1016/j.biosystems.2026.105755) — *BioSystems* (2026). \[[code](https://github.com/todd866/coherence-time-biosystems)\]

## Citation

```bibtex
@article{todd2026codeformation,
  author  = {Todd, Ian},
  title   = {Code Formation at Dimensional Boundaries: Finite-Bandwidth Coordination and Anticipation in Biology},
  journal = {BioSystems},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

MIT License (see [LICENSE](LICENSE)).
