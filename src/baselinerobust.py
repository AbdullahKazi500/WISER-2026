#!/usr/bin/env python3
"""
Robust classical baselines for dual SLA/ASA staffing
====================================================
1. Exact enumeration over shift counts n_s ∈ {0..Nmax}  (tiny: 5^3 = 125)
2. Exact ILP (PuLP/CBC) on unary bits with linear coverage + cost
3. Greedy + multi-start local search
4. Fair comparison vs PCE hybrid (from prior runs)

Scoring rule: dual-feasible (SLA≥80%, ASA≤25s) then minimum cost.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time, json, os
from math import exp, ceil
from itertools import product
import pulp

OUT = "/home/workdir/artifacts/callcenter"
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

TARGET_SL, TARGET_ASA = 0.80, 25.0
AHT, INTERVAL_SEC = 300.0, 1800.0
COST_BASE = 160.0
N_SHIFTS, NMAX = 3, 4
COVER = np.zeros((N_SHIFTS, 12), dtype=int)
COVER[0, 0:8] = 1
COVER[1, 3:11] = 1
COVER[2, 6:12] = 1
SHIFT_COST = np.array([COST_BASE, COST_BASE * 1.05, COST_BASE * 1.1])
SHIFT_NAMES = ["Early", "Mid", "Late"]

print("=" * 72)
print("ROBUST CLASSICAL BASELINES — dual SLA + ASA staffing")
print("=" * 72)

# ---- Erlang / demand ----
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
    return dict(
        n=n.tolist(), cost=cost, sla=sla, asa=asa, agents=int(n.sum()),
        meets_sla=sla >= TARGET_SL - 1e-3,
        meets_asa=asa <= TARGET_ASA + 1e-3,
        meets_both=(sla >= TARGET_SL - 1e-3) and (asa <= TARGET_ASA + 1e-3),
        coverage=cov.tolist(), interval_sla=sls, interval_asa=asas,
    )

# =============================================================================
# 1. Exact enumeration over n_s ∈ {0,...,Nmax}
# =============================================================================
def exact_enumeration(demand, R_sla, R_asa):
    """
    Enumerate all (Nmax+1)^N_SHIFTS integer staffing vectors.
    Filter dual-feasible under true Erlang; return min-cost.
    For N_SHIFTS=3, Nmax=4 → 125 candidates — exact ground truth.
    """
    t0 = time.time()
    best = None
    n_feasible = 0
    n_checked = 0
    for combo in product(range(NMAX + 1), repeat=N_SHIFTS):
        n_checked += 1
        plan = evaluate_plan(combo, demand)
        if plan["meets_both"]:
            n_feasible += 1
            if best is None or plan["cost"] < best["cost"] - 1e-9:
                best = plan
    runtime = time.time() - t0
    if best is None:
        # fallback: min cost among SLA-only, then any
        for combo in product(range(NMAX + 1), repeat=N_SHIFTS):
            plan = evaluate_plan(combo, demand)
            if plan["meets_sla"] and (best is None or plan["cost"] < best["cost"]):
                best = plan
        if best is None:
            best = evaluate_plan((NMAX,) * N_SHIFTS, demand)
        best["tag"] = "NO_DUAL"
    else:
        best["tag"] = "EXACT_DUAL"
    best["time"] = runtime
    best["n_checked"] = n_checked
    best["n_dual_feasible"] = n_feasible
    return best

# =============================================================================
# 2. Exact ILP on unary bits — coverage constraints linearized via R tables
#    Note: true Erlang is nonlinear; ILP uses R_sla/R_asa as hard linear cover.
#    Then we re-score with true Erlang (same as quantum post-filter).
# =============================================================================
def exact_ilp(demand, R_sla, R_asa):
    """
    Minimize sum c_s n_s
    s.t. cov_t >= max(R_sla[t], R_asa[t]) for all t
         n_s = sum_k x_{s,k}, x binary, n_s <= Nmax
    Re-evaluate with Erlang; if dual fail, escalate R and re-solve once.
    """
    t0 = time.time()
    R = np.maximum(R_sla, R_asa).astype(float)

    def solve_with_R(R_use):
        model = pulp.LpProblem("staffing_ilp", pulp.LpMinimize)
        x = {(s, k): pulp.LpVariable(f"x_{s}_{k}", cat="Binary")
             for s in range(N_SHIFTS) for k in range(NMAX)}
        n = {s: pulp.lpSum(x[s, k] for k in range(NMAX)) for s in range(N_SHIFTS)}
        model += pulp.lpSum(SHIFT_COST[s] * n[s] for s in range(N_SHIFTS))
        for t in range(len(demand)):
            cov_t = pulp.lpSum(COVER[s, t] * n[s] for s in range(N_SHIFTS))
            model += cov_t >= float(R_use[t]), f"cover_{t}"
        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
        if pulp.LpStatus[model.status] != "Optimal":
            return None
        n_sol = np.array([int(round(pulp.value(n[s]) or 0)) for s in range(N_SHIFTS)])
        return n_sol

    n_sol = solve_with_R(R)
    if n_sol is None:
        n_sol = np.array([NMAX] * N_SHIFTS)
    plan = evaluate_plan(n_sol, demand)

    # If dual fails under true Erlang, bump R and re-solve
    if not plan["meets_both"]:
        R2 = R + 1
        n_sol2 = solve_with_R(R2)
        if n_sol2 is not None:
            plan2 = evaluate_plan(n_sol2, demand)
            if plan2["meets_both"] or plan2["cost"] >= plan["cost"]:
                if plan2["meets_both"]:
                    plan = plan2

    # Final safety: if still not dual, search neighbors by +1 agent
    if not plan["meets_both"]:
        n = np.array(plan["n"], dtype=int)
        for _ in range(15):
            cov = coverage_from_n(n).astype(float)
            asas = [asa_seconds(int(cov[t]), int(demand[t])) for t in range(len(demand))]
            sls = [service_level(int(cov[t]), int(demand[t])) for t in range(len(demand))]
            if all(s >= TARGET_SL - 1e-3 for s in sls) and max(asas) <= TARGET_ASA:
                break
            # fix worst interval
            if max(asas) > TARGET_ASA:
                t = int(np.argmax(asas))
            else:
                t = int(np.argmin(sls))
            best_s = next((s for s in range(N_SHIFTS) if n[s] < NMAX and COVER[s, t]), None)
            if best_s is None: break
            n[best_s] += 1
        plan = evaluate_plan(n, demand)

    plan["time"] = time.time() - t0
    plan["tag"] = "ILP"
    return plan

# =============================================================================
# 3. Greedy + multi-start local search
# =============================================================================
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

def repair_to_dual(n, demand, R_sla):
    n = np.asarray(n, dtype=int).copy()
    for _ in range(30):
        plan = evaluate_plan(n, demand)
        if plan["meets_both"]:
            return n
        cov = coverage_from_n(n).astype(float)
        asas = [asa_seconds(int(cov[t]), int(demand[t])) for t in range(len(demand))]
        sls = [service_level(int(cov[t]), int(demand[t])) for t in range(len(demand))]
        if max(asas) > TARGET_ASA:
            t = int(np.argmax(asas))
        else:
            t = int(np.argmin(sls))
        best_s = next((s for s in range(N_SHIFTS) if n[s] < NMAX and COVER[s, t]), None)
        if best_s is None: break
        n[best_s] += 1
    return n

def local_search(n0, demand, max_iter=50):
    """
    First-improvement: try −1 agent on a shift if still dual-feasible (cost down),
    else +1 on cheapest shift that repairs dual.
    """
    n = np.asarray(n0, dtype=int).copy()
    n = repair_to_dual(n, demand, None)
    best = evaluate_plan(n, demand)
    for _ in range(max_iter):
        improved = False
        # try remove agent from each shift
        for s in range(N_SHIFTS):
            if n[s] <= 0: continue
            trial = n.copy(); trial[s] -= 1
            plan = evaluate_plan(trial, demand)
            if plan["meets_both"] and plan["cost"] < best["cost"] - 0.5:
                n, best, improved = trial, plan, True
                break
        if improved:
            continue
        # try move: −1 on expensive, +1 on cheap covering shortfall
        break
    return best

def classical_greedy(demand, R_sla, R_asa):
    t0 = time.time()
    n = greedy_warmstart(R_sla, R_asa)
    n = repair_to_dual(n, demand, R_sla)
    plan = evaluate_plan(n, demand)
    plan["time"] = time.time() - t0
    plan["tag"] = "GREEDY"
    return plan

def multi_start_local(demand, R_sla, R_asa, seeds=8):
    t0 = time.time()
    best = None
    for i in range(seeds):
        rng = np.random.default_rng(100 + i)
        if i == 0:
            n0 = greedy_warmstart(R_sla, R_asa)
        else:
            n0 = rng.integers(0, NMAX + 1, size=N_SHIFTS)
        plan = local_search(n0, demand)
        if best is None or (
            plan["meets_both"] and not best["meets_both"]
        ) or (
            plan["meets_both"] == best["meets_both"] and plan["cost"] < best["cost"]
        ):
            best = plan
    best["time"] = time.time() - t0
    best["tag"] = "MULTI_LS"
    return best

# =============================================================================
# RUN all classical + load PCE result if available
# =============================================================================
demand = make_demand(7)
R_sla = np.array([required_agents(a, target_sl=TARGET_SL) for a in demand])
R_asa = np.array([required_agents(a, target_asa=TARGET_ASA) for a in demand])
print(f"Demand sum={demand.sum()}  R_SLA max={R_sla.max()}  R_ASA max={R_asa.max()}")
print(f"State space for exact enum: {(NMAX+1)**N_SHIFTS} staffing vectors\n")

g = classical_greedy(demand, R_sla, R_asa)
print(f"Greedy:          n={g['n']}  cost=${g['cost']:.0f}  SLA={g['sla']:.1%}  ASA={g['asa']:.1f}s  both={g['meets_both']}  t={g['time']:.4f}s")

ls = multi_start_local(demand, R_sla, R_asa)
print(f"Multi-start LS:  n={ls['n']}  cost=${ls['cost']:.0f}  SLA={ls['sla']:.1%}  ASA={ls['asa']:.1f}s  both={ls['meets_both']}  t={ls['time']:.4f}s")

ilp = exact_ilp(demand, R_sla, R_asa)
print(f"ILP (CBC):       n={ilp['n']}  cost=${ilp['cost']:.0f}  SLA={ilp['sla']:.1%}  ASA={ilp['asa']:.1f}s  both={ilp['meets_both']}  t={ilp['time']:.4f}s")

ex = exact_enumeration(demand, R_sla, R_asa)
print(f"Exact enum:      n={ex['n']}  cost=${ex['cost']:.0f}  SLA={ex['sla']:.1%}  ASA={ex['asa']:.1f}s  both={ex['meets_both']}  "
      f"dual_pool={ex['n_dual_feasible']}/{ex['n_checked']}  t={ex['time']:.4f}s")

# PCE from prior JSON if present
pce_cost = pce_sla = pce_asa = None
pce_both = None
pce_n = None
for path in [f"{OUT}/pce_staffing_results.json", f"{OUT}/pce_notebook_results.json"]:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        # prefer post-processed
        if "single" in data and "pce_post" in data["single"]:
            pp = data["single"]["pce_post"]
            pce_cost, pce_sla, pce_asa = pp["cost"], pp["sla"], pp["asa"]
            pce_both, pce_n = pp["meets_both"], pp.get("n")
        elif "pce_post" in data:
            pp = data["pce_post"]
            pce_cost, pce_sla, pce_asa = pp["cost"], pp["sla"], pp["asa"]
            pce_both, pce_n = pp["meets_both"], pp.get("n")
        break

# Multi-seed classical robustness
print("\nMulti-seed classical baselines...")
seed_rows = []
for seed in [7, 11, 21, 42, 99]:
    d = make_demand(seed)
    rs = np.array([required_agents(a, target_sl=TARGET_SL) for a in d])
    ra = np.array([required_agents(a, target_asa=TARGET_ASA) for a in d])
    gv = classical_greedy(d, rs, ra)
    lv = multi_start_local(d, rs, ra)
    iv = exact_ilp(d, rs, ra)
    ev = exact_enumeration(d, rs, ra)
    seed_rows.append(dict(
        seed=seed,
        greedy_cost=gv["cost"], greedy_both=gv["meets_both"],
        ls_cost=lv["cost"], ls_both=lv["meets_both"],
        ilp_cost=iv["cost"], ilp_both=iv["meets_both"],
        exact_cost=ev["cost"], exact_both=ev["meets_both"],
        exact_n=ev["n"], dual_pool=ev["n_dual_feasible"],
    ))
    print(f"  seed={seed}: exact=${ev['cost']:.0f} both={ev['meets_both']} | "
          f"ILP=${iv['cost']:.0f} | LS=${lv['cost']:.0f} | G=${gv['cost']:.0f}")

vdf = pd.DataFrame(seed_rows)

# =============================================================================
# Comparison table + plots
# =============================================================================
rows = [
    dict(method="Greedy", cost=g["cost"], sla=g["sla"], asa=g["asa"], both=g["meets_both"],
         agents=g["agents"], time_s=g["time"], optimality="heuristic"),
    dict(method="Multi-start LS", cost=ls["cost"], sla=ls["sla"], asa=ls["asa"], both=ls["meets_both"],
         agents=ls["agents"], time_s=ls["time"], optimality="heuristic"),
    dict(method="ILP (CBC)", cost=ilp["cost"], sla=ilp["sla"], asa=ilp["asa"], both=ilp["meets_both"],
         agents=ilp["agents"], time_s=ilp["time"], optimality="exact on linear R"),
    dict(method="Exact enum", cost=ex["cost"], sla=ex["sla"], asa=ex["asa"], both=ex["meets_both"],
         agents=ex["agents"], time_s=ex["time"], optimality="EXACT dual-feasible min cost"),
]
if pce_cost is not None:
    rows.append(dict(method="PCE + 1-flip", cost=pce_cost, sla=pce_sla, asa=pce_asa, both=pce_both,
                     agents=sum(pce_n) if pce_n else None, time_s=None, optimality="hybrid quantum"))

comp = pd.DataFrame(rows)
print("\n" + "=" * 72)
print("COMPARISON (score: dual-feasible → min cost)")
print("=" * 72)
print(comp.to_string(index=False))

# Gap to exact
exact_cost = ex["cost"]
print(f"\nGaps to exact dual-feasible optimum (${exact_cost:.0f}):")
for _, r in comp.iterrows():
    if r["both"] and exact_cost > 0:
        gap = r["cost"] - exact_cost
        pct = 100 * gap / exact_cost
        print(f"  {r['method']:16s}  Δ=${gap:+.0f}  ({pct:+.1f}%)")

print("\nMulti-seed exact costs:", vdf["exact_cost"].tolist())
print("Greedy matches exact?", all(vdf["greedy_cost"] == vdf["exact_cost"]))
print("ILP matches exact?", all(vdf["ilp_cost"] == vdf["exact_cost"]))
print("LS matches exact?", all(vdf["ls_cost"] == vdf["exact_cost"]))

# Plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
classical_color, quantum_color = "#2E86AB", "#C0392B"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
methods = comp["method"].tolist()
colors = [classical_color if "PCE" not in m else quantum_color for m in methods]
axes[0].bar(methods, comp["cost"], color=colors, edgecolor="black")
axes[0].axhline(exact_cost, color="#F4A300", ls="--", lw=2, label=f"Exact optimum ${exact_cost:.0f}")
for i, v in enumerate(comp["cost"]):
    axes[0].text(i, v, f"${v:.0f}", ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel("Cost ($)"); axes[0].set_title("Cost by method")
axes[0].tick_params(axis="x", rotation=15); axes[0].legend(fontsize=8)

x = np.arange(len(vdf)); w = 0.2
axes[1].bar(x - 1.5*w, vdf["greedy_cost"], width=w, color="#5DADE2", label="Greedy")
axes[1].bar(x - 0.5*w, vdf["ls_cost"], width=w, color="#2E86AB", label="Multi-LS")
axes[1].bar(x + 0.5*w, vdf["ilp_cost"], width=w, color="#1A5276", label="ILP")
axes[1].bar(x + 1.5*w, vdf["exact_cost"], width=w, color="#F4A300", label="Exact")
axes[1].set_xticks(x); axes[1].set_xticklabels(vdf["seed"])
axes[1].set_xlabel("Seed"); axes[1].set_ylabel("Cost ($)"); axes[1].set_title("Multi-seed classical")
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(f"{OUT}/classical_baseline_comparison.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"\nSaved {OUT}/classical_baseline_comparison.png")

out = dict(
    single=dict(greedy=g, local_search=ls, ilp=ilp, exact=ex, pce_cost=pce_cost),
    multiseed=seed_rows,
    comparison=comp.to_dict(orient="records"),
    exact_cost=exact_cost,
    note="Exact enum is ground truth for dual-feasible min cost on this instance size.",
)
with open(f"{OUT}/classical_baseline_results.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("\n" + "=" * 72)
print("HONEST TAKEAWAY")
print("=" * 72)
print(f"""
Exact dual-feasible optimum: ${exact_cost:.0f}  n={ex['n']}
  Greedy gap:     ${g['cost']-exact_cost:+.0f}
  Multi-start LS: ${ls['cost']-exact_cost:+.0f}
  ILP:            ${ilp['cost']-exact_cost:+.0f}
  PCE (prior):    ${(pce_cost - exact_cost) if pce_cost else float('nan'):+.0f}

On this size (125 staffing vectors), classical exact search is the right baseline.
Greedy alone is weak; ILP + Erlang re-score and multi-start LS are stronger.
Quantum/PCE should be compared to Exact (or ILP), not only to greedy.
""")
print("DONE.")
