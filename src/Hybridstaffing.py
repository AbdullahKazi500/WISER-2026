#!/usr/bin/env python3
"""
PCE (Pauli Correlation Encoding) for Call-Center Staffing
=========================================================
Maps unary staffing bits to a compressed qubit register via multi-body
Pauli correlators, optimizes an EfficientSU2 ansatz, decodes correlations
to hard bits, then classical repair + Erlang dual filter (SLA + ASA).

Defined properly for the staffing problem (not DOM divert bits):
  - Master variables: unary x_{s,k} for agents on shifts
  - Objective: min cost + SLA/ASA shortfall penalties (QUBO energy)
  - PCE: assign each variable to a Pauli string (Z or ZZ); measure <P_i>
  - Soft bit: (1 - tanh(alpha * <P_i>))/2 or sign decode
  - Post-process: repair coverage, evaluate true Erlang SLA/ASA
  - Score: dual-feasible then lowest cost
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time, json, os, math
from math import exp, ceil
from collections import defaultdict
from itertools import combinations
from scipy.optimize import minimize

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

OUT = "/home/workdir/artifacts/callcenter"
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

TARGET_SL, TARGET_ASA = 0.80, 25.0
AHT, INTERVAL_SEC = 300.0, 1800.0
COST_BASE = 160.0
N_SHIFTS, NMAX = 3, 4
N_VARS = N_SHIFTS * NMAX  # 12 master variables
COVER = np.zeros((N_SHIFTS, 12), dtype=int)
COVER[0, 0:8] = 1
COVER[1, 3:11] = 1
COVER[2, 6:12] = 1
SHIFT_COST = np.array([COST_BASE, COST_BASE * 1.05, COST_BASE * 1.1])

# PCE settings
PCE_DEGREE = 2
ANSATZ_REPS = 2
N_RESTARTS = 3
COBYLA_MAXITER = 40
ALPHA_DECODE = 2.0

print("=" * 72)
print("PCE FOR CALL-CENTER STAFFING (dual SLA + ASA)")
print("=" * 72)
print(f"Master variables: {N_VARS}  |  PCE degree: {PCE_DEGREE}")

# ---- Erlang / demand / eval ----
def erlang_c(n, A):
    if n <= 0 or n <= A: return 1.0
    try:
        rho = A / n; B = 1.0
        for k in range(1, n + 1): B = 1.0 + (k / A) * B
        return 1.0 / (1.0 + (1.0 - rho) * (B - 1.0) / rho) if rho > 0 else 0.0
    except Exception:
        return 1.0

def service_level(n, arr, tau=20.0):
    if arr <= 0: return 1.0
    if n <= 0: return 0.0
    A = (arr / INTERVAL_SEC) * AHT
    if n <= A: return 0.0
    pw = erlang_c(n, A)
    return float(max(0, min(1, 1 - pw * exp(-(n - A) * tau / AHT))))

def asa_seconds(n, arr):
    if arr <= 0: return 0.0
    if n <= 0: return 999.0
    A = (arr / INTERVAL_SEC) * AHT
    if n <= A: return 999.0
    return float(erlang_c(n, A) * AHT / (n - A))

def required_agents(arr, target_sl=None, target_asa=None, max_n=40):
    if arr <= 0: return 0
    n = max(1, int(ceil(arr * AHT / INTERVAL_SEC)))
    while n <= max_n:
        ok1 = True if target_sl is None else service_level(n, arr) >= target_sl - 1e-6
        ok2 = True if target_asa is None else asa_seconds(n, arr) <= target_asa + 1e-6
        if ok1 and ok2: return n
        n += 1
    return max_n

def make_demand(seed=7):
    rng = np.random.default_rng(seed)
    base = np.array([5, 9, 14, 18, 22, 20, 16, 15, 18, 16, 11, 6], dtype=float)
    return np.maximum(1, np.round(base * rng.uniform(0.92, 1.08, 12))).astype(int)

def coverage_from_n(n):
    return COVER.T @ np.asarray(n, dtype=int)

def evaluate_plan(n, demand):
    n = np.asarray(n, dtype=int)
    cov = coverage_from_n(n)
    cost = float(np.dot(SHIFT_COST, n))
    sls, asas, calls = [], [], []
    for t, arr in enumerate(demand):
        a = int(cov[t])
        sls.append(service_level(a, int(arr)))
        asas.append(asa_seconds(a, int(arr)))
        calls.append(int(arr))
    w = np.maximum(np.array(calls, float), 1e-6)
    sla = float(np.average(sls, weights=w))
    asa = float(np.average(asas, weights=w))
    return dict(n=n.tolist(), cost=cost, sla=sla, asa=asa, agents=int(n.sum()),
                meets_sla=sla >= TARGET_SL - 1e-3, meets_asa=asa <= TARGET_ASA + 1e-3,
                meets_both=(sla >= TARGET_SL - 1e-3) and (asa <= TARGET_ASA + 1e-3),
                coverage=cov.tolist())

def greedy_warmstart(R_sla, R_asa):
    n = np.zeros(N_SHIFTS, dtype=int)
    rem = np.maximum(R_sla, R_asa).astype(float).copy()
    for _ in range(int(rem.sum()) + 5):
        if rem.max() <= 0: break
        best_s, best_sc = None, -1e99
        for s in range(N_SHIFTS):
            if n[s] >= NMAX: continue
            sc = (COVER[s] @ (rem > 0).astype(float)) / (SHIFT_COST[s] + 1e-6)
            if sc > best_sc: best_sc, best_s = sc, s
        if best_s is None or best_sc <= 0: break
        n[best_s] += 1
        rem = np.maximum(0, rem - COVER[best_s])
    return n

def repair_n(n, R_sla):
    n = np.asarray(n, dtype=int).copy()
    cov = coverage_from_n(n).astype(float)
    for _ in range(int(np.maximum(0, R_sla - cov).sum()) + 5):
        short = R_sla - cov
        if short.max() <= 0: break
        t = int(np.argmax(short))
        best_s = next((s for s in range(N_SHIFTS) if n[s] < NMAX and COVER[s, t]), None)
        if best_s is None: break
        n[best_s] += 1
        cov = coverage_from_n(n).astype(float)
    return n

def classical_greedy(demand, R_sla, R_asa):
    n = repair_n(greedy_warmstart(R_sla, R_asa), R_sla)
    cov = coverage_from_n(n).astype(float)
    for _ in range(25):
        asas = [asa_seconds(int(cov[t]), int(demand[t])) for t in range(len(demand))]
        if max(asas) <= TARGET_ASA: break
        t = int(np.argmax(asas))
        best_s = next((s for s in range(N_SHIFTS) if n[s] < NMAX and COVER[s, t]), None)
        if best_s is None: break
        n[best_s] += 1
        cov = coverage_from_n(n).astype(float)
    return evaluate_plan(n, demand)

# ---- QUBO energy on binary vector (for scoring soft/hard bits) ----
def build_qubo_matrix(R_sla, R_asa, P_sla=400.0, P_asa=120.0):
    """Return dense Q (n_vars x n_vars) and linear l such that E = x^T Q x + l·x."""
    n = N_VARS
    Q = np.zeros((n, n))
    l = np.zeros(n)
    def idx(s, k): return s * NMAX + k
    for s in range(N_SHIFTS):
        for k in range(NMAX):
            l[idx(s, k)] += float(SHIFT_COST[s])
    for t in range(len(R_sla)):
        terms = [idx(s, k) for s in range(N_SHIFTS) if COVER[s, t] for k in range(NMAX)]
        R = float(R_sla[t])
        for i1 in terms:
            l[i1] += P_sla * (1 - 2 * R)  # absorb diagonal of shortfall into linear for x^2=x
        for a in range(len(terms)):
            for b in range(a + 1, len(terms)):
                Q[terms[a], terms[b]] += P_sla
                Q[terms[b], terms[a]] += P_sla
    for t in range(len(R_asa)):
        terms = [idx(s, k) for s in range(N_SHIFTS) if COVER[s, t] for k in range(NMAX)]
        R = float(R_asa[t])
        w = 1.0 + 0.4 * (R / max(1.0, float(R_asa.max())))
        for i1 in terms:
            l[i1] += P_asa * w * (1 - 2 * R)
        for a in range(len(terms)):
            for b in range(a + 1, len(terms)):
                Q[terms[a], terms[b]] += P_asa * w
                Q[terms[b], terms[a]] += P_asa * w
    return Q, l

def qubo_energy(x, Q, l):
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x + l @ x)

def bits_to_n(bits):
    n = np.zeros(N_SHIFTS, dtype=int)
    for s in range(N_SHIFTS):
        n[s] = sum(int(bits[s * NMAX + k]) for k in range(NMAX))
    return n

# ---- PCE mapping: variable i -> Pauli correlator on compressed register ----
def minimum_pce_qubits(n_vars, degree=2):
    """Smallest n_q such that number of Z-type correlators up to degree covers n_vars."""
    n_q = 2
    while True:
        capacity = sum(math.comb(n_q, d) for d in range(1, degree + 1))
        if capacity >= n_vars:
            return n_q
        n_q += 1
        if n_q > n_vars:
            return n_vars

def build_pce_mapping(n_vars, n_qubits, degree=2):
    """
    Assign each master variable to a unique Pauli Z-string of weight <= degree.
    Returns list of SparsePauliOp observables (one per variable).
    """
    strings = []
    # weight-1
    for i in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[i] = "Z"
        strings.append("".join(pauli))
    # weight-2
    if degree >= 2:
        for i, j in combinations(range(n_qubits), 2):
            pauli = ["I"] * n_qubits
            pauli[i] = "Z"
            pauli[j] = "Z"
            strings.append("".join(pauli))
    strings = strings[:n_vars]
    while len(strings) < n_vars:
        # fallback: reuse Z on qubit 0 with phase labels (still unique ops via index)
        strings.append("Z" + "I" * (n_qubits - 1))
    observables = [SparsePauliOp.from_list([(s, 1.0)]) for s in strings]
    mapping = pd.DataFrame({
        "var_index": list(range(n_vars)),
        "shift": [i // NMAX for i in range(n_vars)],
        "slot": [i % NMAX for i in range(n_vars)],
        "pauli": strings,
    })
    return observables, mapping

def hard_bits_from_correlations(corr, alpha=ALPHA_DECODE):
    """Map <P_i> ∈ [-1,1] to bit: positive corr → prefer x=0 (Z=+1 often), use sign."""
    # Convention: high positive <Z> → bit 0, negative → bit 1 (common PCE decode)
    bits = tuple(0 if c >= 0 else 1 for c in corr)
    return bits

def soft_bits_from_correlations(corr, alpha=ALPHA_DECODE):
    # soft ∈ (0,1): probability of diversion/agent-on
    return 0.5 * (1.0 - np.tanh(alpha * np.asarray(corr, dtype=float)))

# ---- PCE hybrid optimizer ----
def run_pce_staffing(demand, R_sla, R_asa):
    Q, l = build_qubo_matrix(R_sla, R_asa)
    n_qubits = minimum_pce_qubits(N_VARS, PCE_DEGREE)
    observables, mapping = build_pce_mapping(N_VARS, n_qubits, PCE_DEGREE)
    print(f"PCE compression: {N_VARS} vars → {n_qubits} qubits "
          f"({100*(1 - n_qubits/N_VARS):.1f}% reduction)")
    print(mapping.head(12).to_string(index=False))

    ansatz = efficient_su2(n_qubits, reps=ANSATZ_REPS, entanglement="linear")
    n_params = ansatz.num_parameters
    estimator = AerEstimator()

    def correlator_values(theta):
        theta = np.asarray(theta, dtype=float)
        # Batch estimate all observables
        circuits = [ansatz] * len(observables)
        params = [theta] * len(observables)
        # EstimatorV2 API
        try:
            pubs = [(ansatz, obs, theta) for obs in observables]
            result = estimator.run(pubs).result()
            vals = np.array([float(np.real(r.data.evs)) for r in result], dtype=float)
        except Exception:
            # fallback: sequential
            vals = []
            for obs in observables:
                r = estimator.run([(ansatz, obs, theta)]).result()
                vals.append(float(np.real(r[0].data.evs)))
            vals = np.array(vals, dtype=float)
        return vals

    def master_loss(theta):
        corr = correlator_values(theta)
        soft = soft_bits_from_correlations(corr)
        # energy of soft relaxation + small barrier on soft bounds
        e = qubo_energy(soft, Q, l)
        return e

    best_record = None
    history = []
    t0 = time.time()

    for restart in range(N_RESTARTS):
        rng = np.random.default_rng(42 + 100 * restart)
        theta0 = rng.uniform(0, 2 * np.pi, size=n_params)
        # warm-ish: bias toward greedy bitstring via small random around zeros
        if restart == 0:
            theta0 = rng.normal(0, 0.3, size=n_params)

        def loss_tracked(th):
            val = master_loss(th)
            history.append(float(val))
            return val

        res = minimize(loss_tracked, theta0, method="COBYLA",
                       options={"maxiter": COBYLA_MAXITER, "tol": 1e-4})
        theta = np.asarray(res.x, dtype=float)
        corr = correlator_values(theta)
        bits = hard_bits_from_correlations(corr)
        n = repair_n(bits_to_n(bits), R_sla)
        # ASA push
        cov = coverage_from_n(n).astype(float)
        for _ in range(20):
            asas = [asa_seconds(int(cov[t]), int(demand[t])) for t in range(len(demand))]
            if max(asas) <= TARGET_ASA: break
            t = int(np.argmax(asas))
            best_s = next((s for s in range(N_SHIFTS) if n[s] < NMAX and COVER[s, t]), None)
            if best_s is None: break
            n[best_s] += 1
            cov = coverage_from_n(n).astype(float)
        plan = evaluate_plan(n, demand)
        plan["bits"] = bits
        plan["corr"] = corr.tolist()
        plan["theta"] = theta.tolist()
        plan["restart"] = restart
        plan["master_loss"] = float(res.fun)
        if best_record is None or (
            plan["meets_both"] and not best_record["meets_both"]
        ) or (
            plan["meets_both"] == best_record["meets_both"] and plan["cost"] < best_record["cost"]
        ):
            best_record = plan

    runtime = time.time() - t0
    best_record["time"] = runtime
    best_record["n_qubits"] = n_qubits
    best_record["n_vars"] = N_VARS
    best_record["history"] = history
    best_record["mapping"] = mapping
    return best_record

# =============================================================================
# RUN + VALIDATE
# =============================================================================
demand = make_demand(7)
R_sla = np.array([required_agents(a, target_sl=TARGET_SL) for a in demand])
R_asa = np.array([required_agents(a, target_asa=TARGET_ASA) for a in demand])
print(f"\nDemand sum={demand.sum()}  R_SLA peak={R_sla.max()}  R_ASA peak={R_asa.max()}")

g = classical_greedy(demand, R_sla, R_asa)
print(f"Greedy: cost=${g['cost']:.0f} SLA={g['sla']:.1%} ASA={g['asa']:.1f}s both={g['meets_both']}")

print("\nRunning PCE hybrid...")
pce = run_pce_staffing(demand, R_sla, R_asa)
print(f"PCE:    cost=${pce['cost']:.0f} SLA={pce['sla']:.1%} ASA={pce['asa']:.1f}s both={pce['meets_both']} "
      f"qubits={pce['n_qubits']} t={pce['time']:.2f}s")

# One-flip local search post-process on PCE bits
def one_flip(bits, demand, R_sla):
    best_n = repair_n(bits_to_n(bits), R_sla)
    best_plan = evaluate_plan(best_n, demand)
    cur = list(bits)
    improved = True
    while improved:
        improved = False
        for i in range(len(cur)):
            trial = cur.copy()
            trial[i] = 1 - trial[i]
            n = repair_n(bits_to_n(trial), R_sla)
            plan = evaluate_plan(n, demand)
            better = (
                (plan["meets_both"] and not best_plan["meets_both"]) or
                (plan["meets_both"] == best_plan["meets_both"] and plan["cost"] < best_plan["cost"] - 0.5)
            )
            if better:
                best_plan = plan
                cur = trial
                improved = True
                break
    return best_plan

pce_pp = one_flip(pce["bits"], demand, R_sla)
print(f"PCE+1flip: cost=${pce_pp['cost']:.0f} SLA={pce_pp['sla']:.1%} ASA={pce_pp['asa']:.1f}s both={pce_pp['meets_both']}")

# Multi-seed validation
print("\nMulti-seed validation...")
seed_rows = []
for seed in [7, 11, 21, 42, 99]:
    d = make_demand(seed)
    rs = np.array([required_agents(a, target_sl=TARGET_SL) for a in d])
    ra = np.array([required_agents(a, target_asa=TARGET_ASA) for a in d])
    gv = classical_greedy(d, rs, ra)
    pv = run_pce_staffing(d, rs, ra)
    pp = one_flip(pv["bits"], d, rs)
    beat = pp["meets_both"] and gv["meets_both"] and pp["cost"] < gv["cost"] - 0.5
    seed_rows.append(dict(seed=seed, g_cost=gv["cost"], pce_cost=pp["cost"],
                          g_both=gv["meets_both"], pce_both=pp["meets_both"], beat=beat,
                          g_sla=gv["sla"], pce_sla=pp["sla"], g_asa=gv["asa"], pce_asa=pp["asa"]))
    print(f"  seed={seed}: G=${gv['cost']:.0f} both={gv['meets_both']} | "
          f"PCE=${pp['cost']:.0f} both={pp['meets_both']} beat={beat}")

vdf = pd.DataFrame(seed_rows)
n_both = int(vdf["pce_both"].sum())
n_beat = int(vdf["beat"].sum())

print("\n" + "=" * 72)
print("VERIFICATION SUMMARY")
print("=" * 72)
print(f"PCE dual-feasible: {n_both}/{len(vdf)}")
print(f"PCE strictly cheaper than greedy: {n_beat}/{len(vdf)}")
if n_both == len(vdf) and n_beat > 0:
    print("CONFIRMED: PCE hybrid meets SLA+ASA and can beat greedy on cost.")
elif n_both > 0:
    print("PARTIAL: dual-feasible on some seeds.")
else:
    print("NOT dual-feasible across seeds — check encoding/decode.")

comparison = pd.DataFrame({
    "Method": ["Greedy", "PCE raw", "PCE + 1-flip"],
    "Cost": [g["cost"], pce["cost"], pce_pp["cost"]],
    "SLA": [g["sla"], pce["sla"], pce_pp["sla"]],
    "ASA": [g["asa"], pce["asa"], pce_pp["asa"]],
    "Both": [g["meets_both"], pce["meets_both"], pce_pp["meets_both"]],
    "Qubits": ["-", pce["n_qubits"], pce["n_qubits"]],
})
print("\n", comparison.to_string(index=False))

out = dict(
    definition=dict(
        master_variables=N_VARS,
        encoding="unary agents x_{s,k}",
        pce_degree=PCE_DEGREE,
        compressed_qubits=int(pce["n_qubits"]),
        ansatz=f"efficient_su2 reps={ANSATZ_REPS}",
        decode="sign(<P_i>) → hard bit; soft via tanh",
        postprocess="coverage repair + ASA push + 1-flip",
        score="SLA>=80% and ASA<=25s then min cost",
    ),
    single=dict(greedy=g, pce=pce, pce_post=pce_pp),
    multiseed=seed_rows,
    n_both=n_both, n_beat=n_beat,
)
# strip heavy fields
for k in ["history", "mapping", "theta", "corr"]:
    if k in out["single"]["pce"]:
        del out["single"]["pce"][k]

with open(f"{OUT}/pce_staffing_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved → {OUT}/pce_staffing_results.json")
print("DONE.")
