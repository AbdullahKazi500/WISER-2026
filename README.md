# Hybrid Quantum–Classical Call-Center Staffing Optimization
# WISER Global Quantum+AI Program 2026 — Nestlé Challenge Team Zero dyn

> A reproducible research platform that formulates dual-constrained workforce staffing as a
> binary optimization problem, solves reduced instances with classical, quantum-inspired, and
> gate-based methods, and validates every candidate against **true Erlang-C SLA and ASA** —
> with an exact dual-feasible optimum as ground truth, not a single “beats greedy” headline.

---

## In 60 seconds

**What it is.** A reproducible platform that encodes call-center shift staffing as a slack-free
QUBO (cost + SLA/ASA shortfall penalties), solves it with classical heuristics, exact
enumeration, warm-start CVaR-QAOA, and Pauli Correlation Encoding (PCE), then accepts only plans
that meet **SLA ≥ 80%** and **ASA ≤ 25 s** under simulation.

**What it does not claim.** Exact enumeration (125 staffing vectors on the standard instance)
already finds the dual-feasible minimum cost. **No quantum-advantage claim is made at 5–12
qubits.** The contribution is an honest hybrid pipeline aligned with the scoring rule, a
qubit-efficient PCE path, and a baseline ladder that attributes performance to formulation,
solver, and post-processing separately.

**The three findings that matter:**

| | |
|---|---|
| The dual gate is the score | Official ranking is **dual-feasible → min cost**, not SLA-only and not unconstrained cost |
| Hybrids match exact, not just greedy | Exact dual optimum **$1496** (`n* = [4,3,2]`); PCE + 1-flip and CVaR-QAOA recover it; greedy is **+$176** |
| PCE compresses qubits at equal quality | **12 master variables → 5 qubits** (~58% reduction) at the same dual-feasible cost |

**Where the evidence is.** Executed notebooks under `Notebooks/`, numerical JSON under
results embedded in those runs, the mathematical formulation PDF under `Docs/`, and the Beamer
deck under `Presentation/`. Every cost/SLA/ASA number below traces to those artifacts.

### Claims and their evidence

| Claim | Evidence | Strength |
|---|---|---|
| Dual-feasible exact optimum is $1496 | Exact enum over 125 vectors; `classical_baselines` notebook | Strong on the standard instance |
| Multi-start LS matches exact | Multi-seed table (5/5) in classical baselines | Strong on this size |
| Greedy is dual-feasible but suboptimal | +$176 gap (~12%) vs exact | Strong |
| Linear ILP over-staffs vs Erlang scoring | $2016 vs $1496 | Strong illustration of linear-$R$ mismatch |
| PCE + 1-flip reaches exact dual cost | `pce_staffing` / multi-seed | Strong after post-process; raw PCE alone often ≈ greedy |
| CVaR-QAOA hybrid reaches exact dual cost | Hybrid / CVaR notebooks | Strong with repair + best-of-N |
| PCE uses fewer qubits at same quality | 12 → 5 qubits mapping table | Strong for compression; not a runtime win on Aer MPS |
| Quantum is required for this optimum | Random / multi-start classical with same repair | **Not supported** at 12 qubits — stated deliberately |

---

## Status

This README describes a completed challenge-oriented submission, not an aspirational plan.
Where intermediate drafts emphasized “beats greedy,” later measurement against **exact dual
enumeration** corrected the headline: hybrids match the exact dual-feasible minimum cost;
greedy is the weak baseline. That correction is stated explicitly rather than silently dropped.

**Implemented and verified:**

- Synthetic interval demand (TOD-style profile + seeded noise) and Erlang-C requirement tables
  \(R^{\mathrm{SLA}}\), \(R^{\mathrm{ASA}}\).
- Unary binary encoding of agents per shift; coverage incidence with breaks.
- Slack-free dual QUBO (staffing cost + shortfall-squared SLA/ASA penalties) and Ising map.
- Classical ladder: greedy, multi-start local search, linear ILP (PuLP/CBC), **exact
  enumeration** as ground truth under the scoring rule.
- Warm-start CVaR-QAOA (Aer MPS / Estimator) with classical repair and dual Erlang gate.
- Pauli Correlation Encoding (degree-2 Z/ZZ map, EfficientSU2, soft energy, hard decode,
  1-flip post-process).
- Optional dual-averaging / Lagrangian-style outer updates on residual shortfalls.
- Multi-seed robustness checks; head-to-head cost / SLA / ASA / coverage plots.
- Mathematical formulation (LaTeX/PDF), 10-page Beamer deck, 15-page technical report.
- Theory + executed notebooks for classical, CVaR-QAOA, PCE, hybrid, MOQA experiments, and
  quantum diagnostics.

**Deferred / out of scope for this submission:** live QPU runs, production WFM API integration,
real multi-skill historical traces, Optuna hyperparameter search, Docker image. See
[Scope and Deferred Work](#scope-and-deferred-work).

---

## Table of Contents

1. [Challenge Context](#challenge-context)
2. [Scoring Rule and Diagnostic Ladder](#scoring-rule-and-diagnostic-ladder)
3. [Quick Start](#quick-start)
4. [Results](#results)
5. [Methods](#methods)
6. [Repository Layout](#repository-layout)
7. [Notebooks](#notebooks)
8. [Scope and Deferred Work](#scope-and-deferred-work)
9. [Known Limitations](#known-limitations)
10. [Reproducibility](#reproducibility)
11. [Judging Alignment](#judging-alignment)
12. [Citation](#citation)
13. [License](#license)

---

## Challenge Context

Call-center staffing is a large-scale combinatorial problem: forecast interval demand, assign
agents to shifts under break and OT rules, and meet service targets at minimum cost. The
challenge asks for quantum or quantum-inspired methods benchmarked against transparent
classical baselines, with emphasis on:

- mathematical formulation (binary variables, constraints, objective);
- quantum-compatible (QUBO / Ising) reformulation;
- classical validation and simulation of SLA, ASA, utilization;
- manager-facing trade-offs (cost vs service);
- speed, optimality, and scalability.

| Challenge theme | This repository |
|---|---|
| Dual service targets | Erlang-C SLA + ASA; hard dual gate after decode |
| Optimization formulation | Unary agents, coverage, slack-free dual QUBO |
| Classical baselines | Greedy, multi-start LS, linear ILP, **exact enum** |
| Gate-based quantum | Warm-start CVaR-QAOA (Qiskit Aer) |
| Qubit-efficient quantum | PCE (12 → 5 qubits) |
| Simulation of outcomes | Interval SLA, ASA, coverage, cost |
| Scalability analysis | PCE correlator capacity; decomposition path in report |
| Communication | Beamer deck, technical report, math formulation PDF |

---

## Scoring Rule and Diagnostic Ladder

### Official score

\[
\textbf{Accept only if }\mathrm{SLA}\ge 80\%\ \textbf{and}\ \mathrm{ASA}\le 25\,\mathrm{s};
\quad\textbf{then minimize}\ \sum_s c_s n_s.
\]

Linear covers \(\mathrm{cov}_t \ge R_t\) are used inside the QUBO; **acceptance always uses
nonlinear Erlang simulation**.

### Four-stage diagnostic ladder (staffing analogue)

A decoded bitstring can fail for independent reasons. Reporting only “cost vs greedy” conflates
them. This project separates:

| Stage | Question | What it needs |
|---|---|---|
| **A — Representability** | Can the unary shift model express a dual-feasible plan at all under \(N_{\max}\)? | Feasibility of the integer box |
| **B — Formulation fidelity** | Is the dual-feasible minimum-cost plan a low-energy state of the QUBO (after penalties)? | Exact enum + QUBO energy on \(\mathbf{n}^\star\) |
| **C — Solver reach** | Did this solver (with its shot budget / restarts) reach a dual-feasible optimum-cost plan after repair? | Exact cost as reference |
| **D — Operational quality** | Simulated SLA, ASA, utilization, understaffed intervals | Always computable |

**Measured on the standard instance (\(S=3\), \(N_{\max}=4\), \(T=12\)):**

| Stage | Result |
|---|---|
| A | Dual-feasible plans exist (8/125 vectors in one full enum) |
| B / D | Exact dual optimum cost **$1496**, SLA ≈ 93.7%, ASA ≈ 12.1 s |
| C (multi-start LS) | Matches exact |
| C (PCE + 1-flip) | Matches exact |
| C (CVaR hybrid) | Matches exact |
| C (greedy) | Dual-feasible but **+$176** |

Attribution examples: “post-process required (raw PCE ≈ greedy)”, “linear ILP over-cover”,
“no failure: dual cost matches exact”.

---

## Quick Start

```bash
# Environment (Python 3.10+ recommended)
pip install -r requirements.txt
# Typical pins used in runs: numpy, pandas, scipy, matplotlib, pulp, qiskit, qiskit-aer

# Classical ladder (exact ground truth)
python src/classical_baseline_robust.py   # or open Notebooks/classical_baselines.ipynb

# PCE hybrid
python src/pce_staffing.py                # or Notebooks/pce / hybrid notebooks

# Full hybrid CVaR path
python src/hybridstaffingfull.py          # mirrors hybrid_detailed notebooks
```

Open executed notebooks under `Notebooks/` for plots and full narratives. PDFs under `Docs/`
and `Presentation/` summarize math and slides.

---

## Results

### Standard-instance scorecard

| Method | Cost | Gap to exact | Dual-feasible | Qubits |
|---|---:|---:|---|---|
| **Exact enumeration** | **$1496** | $0 | Yes | — |
| Multi-start local search | **$1496** | $0 | Yes | — |
| **PCE + 1-flip** | **$1496** | $0 | Yes | **5** |
| CVaR-QAOA hybrid | **$1496** | $0 | Yes | 12 |
| Greedy | $1672 | +$176 | Yes | — |
| Linear ILP (CBC) | $2016 | +$520 | Yes | — |

Exact plan: \(n^\star = [4, 3, 2]\) (Early / Mid / Late).

### Multi-seed (demand seeds 7, 11, 21, 42, 99)

- Multi-start LS matches exact dual cost pattern.
- Post-processed hybrids remain dual-feasible and track exact.
- Greedy stays dual-feasible but consistently higher cost.

### Interpretation (honest)

Matching **exact** is the correct optimality claim. “Beats greedy” is true but incomplete: a
strong classical heuristic also reaches exact on this size. PCE’s distinctive result is **equal
quality at fewer qubits**. Claims that a quantum circuit was *necessary* for $1496 are **not
supported** at 5–12 qubits when the same repair budget is given to classical multi-start /
random-restart controls.

---

## Methods

### Formulation (short)

- **Variables:** unary \(x_{s,k}\in\{0,1\}\), \(n_s=\sum_k x_{s,k}\), \(\mathrm{cov}_t=\sum_s C_{s,t}n_s\).
- **Hard score:** dual Erlang filter + minimize \(\sum_s c_s n_s\).
- **QUBO:** cost + \(P_{\mathrm{SLA}}\sum_t(\mathrm{cov}_t-R_t^{\mathrm{SLA}})^2\) + ASA analogue (slack-free).
- **Ising:** \(x=(1-Z)/2\).

Full derivation: `Docs/staffing_mathematical_formulation.pdf` and the technical report.

### Classical

| Solver | Role |
|---|---|
| Exact enum | Ground truth under scoring rule (125 vectors) |
| Multi-start LS | Strong heuristic; matches exact here |
| Linear ILP | Cover on \(\max(R^{\mathrm{SLA}},R^{\mathrm{ASA}})\); re-score with Erlang |
| Greedy | Fast constructive + repair; weak baseline |

### Quantum / hybrid

| Method | Idea |
|---|---|
| Warm-start CVaR-QAOA | \(R_Y\) from classical bits; minimize CVaR of energy tail; sample → repair → dual gate |
| PCE | Map 12 vars → 5 qubits via Z/ZZ correlators; soft energy training; sign decode; 1-flip |
| MOQA experiments | Multi-objective Hamiltonian sketches in dedicated notebook (exploratory) |

### Post-processing (shared)

Coverage repair → ASA push → optional one-flip local search → **hard dual Erlang acceptance**.

---

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── Docs/                          # Math formulation, technical report PDFs / TeX
├── Presentation/                  # Beamer deck (10 frames)
├── Notebooks/
│   ├── Readme.md
│   ├── Notebook_1_Hybrid_Staffing_Verified_executed.ipynb
│   ├── Notebook_2_Quantum_Diagnostics_executed.ipynb
│   ├── classical_baselines.ipynb
│   ├── hybrid_detailed_theory.ipynb
│   ├── hybrid_detailed_executed.ipynb
│   ├── cvar_qaoa_staffing_theory.ipynb
│   ├── cvar_qaoa_staffing_executed.ipynb
│   ├── pce_staffing_theory*.ipynb
│   ├── moqa_staffing_executed.ipynb
│   ├── qaoa_sla_asa__theory_.ipynb
│   └── qaoa_sla_asa_verify_executed.ipynb
└── src/
    ├── hybridstaffingfull.py      # End-to-end hybrid runner
    ├── classical_baseline_robust.py
    ├── pce_staffing.py
    ├── cvar_qaoa_staffing.py
    └── ...                        # helpers, plot scripts, notebook builders
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `Notebook_1_Hybrid_Staffing_Verified_executed` | Full hybrid narrative + exact gap + random-control caveat |
| `Notebook_2_Quantum_Diagnostics_executed` | Circuit / sampling diagnostics companion |
| `classical_baselines` | Exact, ILP, LS, greedy, multi-seed |
| `hybrid_detailed_*` | Theory + executed hybrid pipeline |
| `cvar_qaoa_staffing_*` | CVaR-QAOA theory and executed runs |
| `pce_staffing_*` | PCE encoding, training, compression |
| `moqa_staffing_executed` | Multi-objective quantum approximation experiments |
| `qaoa_sla_asa_*` | Dual-metric QAOA verify path |

Prefer **executed** notebooks for figures and numbers; theory notebooks for derivation without
heavy simulator runtime.

---

## Scope and Deferred Work

Stated plainly:

- **No live IBM / D-Wave QPU jobs** in the scored results — Aer / local simulation only
  (optional fake-backend noise studies may appear in diagnostics notebooks).
- **No production workforce-management API** export.
- **Synthetic demand** is structured but not fit to a confidential call-center trace.
- **Docker / full CI matrix** not required for this submission size.
- **Enterprise scale** (many skills × 48 intervals × large \(N_{\max}\)) is discussed as a
  decomposition path, not solved monolithically here.

If a reader expects any of the above and does not find it, this is why.

---

## Known Limitations

1. **Small exact-verification regime.** Exact enum is practical because \(S\) and \(N_{\max}\)
   are small. Above that, Stage C claims need MIP or high-quality classical heuristics, not
   exhaustive search.

2. **Post-processing is load-bearing.** Raw PCE or low-shot QAOA often need repair / 1-flip /
   best-of-N to reach $1496. Comparing quantum *decode alone* to classical *with repair* is
   not a fair solver comparison.

3. **Linear ILP ≠ dual optimum.** Hard \(R\) covers are conservative relative to nonlinear
   Erlang scoring.

4. **No quantum-advantage claim.** At 5–12 qubits, classical multi-start with the same repair
   budget is competitive. That finding strengthens honesty, not a failure of the submission.

5. **Simulator ≠ device.** Gate counts / depth estimates may use fake backends; they are not
   hardware execution results unless explicitly labeled.

6. **Research prototype.** Not a deployment-ready WFM product.

---

## Reproducibility

- Seeded demand generation and solver RNGs where applicable.
- Executed notebooks capture figures and printed tables from the runs that produced the
  scorecard above.
- Math symbols and penalty structure are fixed in `Docs/staffing_mathematical_formulation.pdf`.
- Typical stack: `numpy`, `scipy`, `pandas`, `matplotlib`, `pulp`, `qiskit`, `qiskit-aer`.

```bash
pip install -r requirements.txt
# Re-run classical ground truth
python src/classical_baseline_robust.py
```

Full quantum notebook re-execution can take longer on CPU-only Aer MPS; use theory notebooks
for offline reading.

---

## Judging Alignment

| Axis | How this repo addresses it |
|---|---|
| **Optimality** | Dual gate + cost; gaps reported **vs exact $1496**, not only vs greedy |
| **Speed** | Exact/LS on the small box are millisecond-scale; hybrid quantum steps ~0.3–2 s in our MPS runs |
| **Scalability** | PCE 12→5 qubits; slack-free encoding; block-decomposition path in the technical report |

Manager-style scenarios (cost / service / preference / resilience) reweight the same model
without dropping the dual acceptance gate.

---

## Citation

Suggested citation:

```text
Hybrid Quantum–Classical Call-Center Staffing under Dual SLA/ASA Constraints
(2026). Challenge submission notebooks and source under this repository.
```

Related challenge context: workforce staffing optimization with quantum/hybrid methods,
evaluated on service quality and cost.

---

## License

See [`LICENSE`](LICENSE). Third-party libraries (NumPy, SciPy, Qiskit, PuLP, etc.) retain their
own licenses.

---

## Disclaimer

This repository is a **research prototype** for an academic/industry challenge. It is not a
production workforce-management system, not a guarantee of hardware speedup, and not a
substitute for site-specific forecasting, labor law, or operational validation. Results apply
to the documented synthetic instances and solver settings.
