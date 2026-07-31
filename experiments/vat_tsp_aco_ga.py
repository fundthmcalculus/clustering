"""ACO & GA for TSP, hot-started from the VAT multi-start consensus (soft prior).

Closes the thread: the VAT sequence-variation study produced a co-adjacency matrix
C[a,b] = fraction of VAT starts in which a,b are consecutive. The 2-opt benchmark
showed C must be used as a **soft prior**, not a hard freeze. So:

  * **ACO** (Ant Colony System, memetic — 2-opt on every ant): initial pheromone
    tau0(a,b) = base * (1 + kappa * C[a,b]) instead of flat. Consensus edges start
    hot; evaporation lets the colony adapt.
  * **GA** (order crossover OX, memetic — 2-opt on every child): seed part of the
    initial population with the VAT orders (one per start) instead of all-random.

Each is run standard-init vs VAT-hot-start under the **same evaluation budget**
(number of tours built + locally optimised), so the curves are a fair anytime
comparison. Quality = % over published optimum; nearest-size EUC_2D TSPLIB.
Reference lines: single VAT+2-opt, and best of the multi-start VAT+2-opt.

Run:  python -m experiments.vat_tsp_aco_ga
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from numba import njit  # noqa: E402

from tribbleclustering import gpu  # noqa: E402
from experiments.vat_tsp_tsplib import (  # noqa: E402
    knn_device,
    nearest_euc_instance,
    optimal_length,
)
from experiments.vat_tsp_dualvat_lk import tour_len  # noqa: E402
from experiments.vat_tsp_2opt_bench import vat_order_nb, two_opt_only  # noqa: E402

if gpu.is_available():
    import cupy as cp

FIG_DIR = Path(__file__).parent / "figures"


# --------------------------------------------------------------------------- #
# consensus prior
# --------------------------------------------------------------------------- #
def build_consensus(D, n_starts):
    """C[a,b] = fraction of VAT starts (evenly spaced) with a,b consecutive."""
    n = D.shape[0]
    starts = np.linspace(0, n - 1, min(n_starts, n), dtype=int)
    C = np.zeros((n, n))
    for s in starts:
        o = vat_order_nb(D, int(s))
        a, b = o[:-1], o[1:]
        np.add.at(C, (a, b), 1.0)
        np.add.at(C, (b, a), 1.0)
    return C / len(starts), starts


# --------------------------------------------------------------------------- #
# ACO — Ant Colony System, memetic (2-opt on every ant)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def aco_run(D, coords, cand, tau, n_ants, n_iter, alpha, beta, rho, elite, ceil, curve):
    n = D.shape[0]
    K = cand.shape[1]
    allow = np.ones(n, np.bool_)
    best_len = 1e18
    best_tour = np.arange(n)
    bufw = np.empty(K, np.float64)
    for it in range(n_iter):
        ib_len = 1e18
        ib_tour = np.arange(n)
        for _ant in range(n_ants):
            visited = np.zeros(n, np.bool_)
            tour = np.empty(n, np.int64)
            start = np.random.randint(n)
            tour[0] = start
            visited[start] = True
            cur = start
            for step in range(1, n):
                wsum = 0.0
                for t in range(K):
                    v = cand[cur, t]
                    if v >= 0 and not visited[v]:
                        w = (tau[cur, v] ** alpha) * (
                            (1.0 / (D[cur, v] + 1e-9)) ** beta
                        )
                        bufw[t] = w
                        wsum += w
                    else:
                        bufw[t] = 0.0
                pick = -1
                if wsum > 0.0:
                    r = np.random.random() * wsum
                    acc = 0.0
                    for t in range(K):
                        if bufw[t] > 0.0:
                            acc += bufw[t]
                            if acc >= r:
                                pick = cand[cur, t]
                                break
                    if pick < 0:
                        for t in range(K - 1, -1, -1):
                            if bufw[t] > 0.0:
                                pick = cand[cur, t]
                                break
                if pick < 0:  # no unvisited candidate: nearest unvisited overall
                    bd = 1e18
                    for v in range(n):
                        if not visited[v] and D[cur, v] < bd:
                            bd = D[cur, v]
                            pick = v
                tour[step] = pick
                visited[pick] = True
                cur = pick
            two_opt_only(tour, coords, cand, ceil, allow, 60)
            L = tour_len(tour, coords, ceil)
            if L < ib_len:
                ib_len = L
                ib_tour = tour.copy()
        # evaporate + deposit (iteration-best and elite global-best)
        for i in range(n):
            for j in range(n):
                tau[i, j] *= 1.0 - rho
        dep = 1.0 / ib_len
        for i in range(n):
            a = ib_tour[i]
            b = ib_tour[(i + 1) % n]
            tau[a, b] += dep
            tau[b, a] += dep
        if ib_len < best_len:
            best_len = ib_len
            best_tour = ib_tour.copy()
        depb = elite / best_len
        for i in range(n):
            a = best_tour[i]
            b = best_tour[(i + 1) % n]
            tau[a, b] += depb
            tau[b, a] += depb
        curve[it] = best_len
    return best_tour, best_len


# --------------------------------------------------------------------------- #
# GA — order crossover (OX), memetic (2-opt on every child)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _ox(p1, p2, out, used):
    n = p1.shape[0]
    a = np.random.randint(n)
    b = np.random.randint(n)
    if a > b:
        a, b = b, a
    for i in range(n):
        used[i] = False
    for i in range(a, b + 1):
        out[i] = p1[i]
        used[p1[i]] = True
    idx = (b + 1) % n
    for k in range(n):
        c = p2[(b + 1 + k) % n]
        if not used[c]:
            out[idx] = c
            used[c] = True
            idx = (idx + 1) % n


@njit(cache=True)
def ga_run(D, coords, knn, pop, n_gen, tsize, pm, ceil, curve):
    P = pop.shape[0]
    n = pop.shape[1]
    allow = np.ones(n, np.bool_)
    used = np.empty(n, np.bool_)
    lens = np.empty(P, np.float64)
    for i in range(P):
        two_opt_only(pop[i], coords, knn, ceil, allow, 60)
        lens[i] = tour_len(pop[i], coords, ceil)
    newpop = np.empty_like(pop)
    child = np.empty(n, np.int64)
    for g in range(n_gen):
        # elitism: carry the current best
        be = 0
        for i in range(1, P):
            if lens[i] < lens[be]:
                be = i
        newpop[0] = pop[be].copy()
        for c in range(1, P):
            # tournament parents
            p1 = np.random.randint(P)
            for _ in range(tsize - 1):
                q = np.random.randint(P)
                if lens[q] < lens[p1]:
                    p1 = q
            p2 = np.random.randint(P)
            for _ in range(tsize - 1):
                q = np.random.randint(P)
                if lens[q] < lens[p2]:
                    p2 = q
            _ox(pop[p1], pop[p2], child, used)
            if np.random.random() < pm:  # segment-reversal mutation
                a = np.random.randint(n)
                b = np.random.randint(n)
                if a > b:
                    a, b = b, a
                lo, hi = a, b
                while lo < hi:
                    child[lo], child[hi] = child[hi], child[lo]
                    lo += 1
                    hi -= 1
            two_opt_only(child, coords, knn, ceil, allow, 60)
            newpop[c] = child.copy()
        for i in range(P):
            pop[i] = newpop[i]
            lens[i] = tour_len(pop[i], coords, ceil)
        bl = lens[0]
        for i in range(1, P):
            if lens[i] < bl:
                bl = lens[i]
        curve[g] = bl
    be = 0
    for i in range(1, P):
        if lens[i] < lens[be]:
            be = i
    return pop[be].copy(), lens[be]


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def run_instance(
    target, budget=6000, n_ants=20, pop=50, n_starts=24, kappa=5.0, seed=0
):
    name, coords, dim = nearest_euc_instance(target)
    opt = optimal_length(name)
    ref = float(opt) if opt else 1.0
    Dg = gpu.pairwise_distances_device(coords, dtype="float32")
    D = cp.asnumpy(Dg)
    knn = knn_device(Dg, 12)
    del Dg
    cp.get_default_memory_pool().free_all_blocks()

    C, starts = build_consensus(D, n_starts)

    def pct(L):
        return 100.0 * (L - ref) / ref

    # references: single VAT+2-opt, and best multi-start VAT+2-opt
    allow = np.ones(dim, np.bool_)
    vat_lens = []
    for s in starts:
        t = vat_order_nb(D, int(s))
        two_opt_only(t, coords, knn, False, allow, 80)
        vat_lens.append(tour_len(t, coords, False))
    ref_single = pct(vat_lens[0])
    ref_multi = pct(min(vat_lens))

    base_tau = 1.0
    n_iter = max(1, budget // n_ants)
    n_gen = max(1, budget // pop)
    out = {"name": name, "n": dim, "ref_single": ref_single, "ref_multi": ref_multi}

    # ---- ACO: flat vs VAT prior ----
    np.random.seed(seed)
    tau_flat = np.full((dim, dim), base_tau)
    cur = np.empty(n_iter)
    t0 = time.perf_counter()
    _, L = aco_run(
        D, coords, knn, tau_flat, n_ants, n_iter, 1.0, 3.0, 0.1, 1.0, False, cur
    )
    out["aco_flat"] = (
        pct(cur.copy()),
        pct(L),
        time.perf_counter() - t0,
        (np.arange(n_iter) + 1) * n_ants,
    )

    np.random.seed(seed)
    tau_vat = base_tau * (1.0 + kappa * C)
    cur = np.empty(n_iter)
    t0 = time.perf_counter()
    _, L = aco_run(
        D, coords, knn, tau_vat, n_ants, n_iter, 1.0, 3.0, 0.1, 1.0, False, cur
    )
    out["aco_vat"] = (
        pct(cur.copy()),
        pct(L),
        time.perf_counter() - t0,
        (np.arange(n_iter) + 1) * n_ants,
    )

    # ---- GA: random population vs VAT-seeded ----
    rng = np.random.default_rng(seed)
    pop_rand = np.stack([rng.permutation(dim) for _ in range(pop)]).astype(np.int64)
    np.random.seed(seed)
    cur = np.empty(n_gen)
    t0 = time.perf_counter()
    _, L = ga_run(D, coords, knn, pop_rand.copy(), n_gen, 4, 0.1, False, cur)
    out["ga_rand"] = (
        pct(cur.copy()),
        pct(L),
        time.perf_counter() - t0,
        (np.arange(n_gen) + 1) * pop,
    )

    pop_vat = pop_rand.copy()
    for k, s in enumerate(starts[:pop]):  # seed as many rows as we have starts
        pop_vat[k] = vat_order_nb(D, int(s))
    np.random.seed(seed)
    cur = np.empty(n_gen)
    t0 = time.perf_counter()
    _, L = ga_run(D, coords, knn, pop_vat, n_gen, 4, 0.1, False, cur)
    out["ga_vat"] = (
        pct(cur.copy()),
        pct(L),
        time.perf_counter() - t0,
        (np.arange(n_gen) + 1) * pop,
    )
    return out


def figure(primary, summary):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    # (a) anytime convergence on the primary instance
    styles = {
        "aco_flat": ("tab:blue", "-", "ACO flat-tau0"),
        "aco_vat": ("tab:blue", "--", "ACO VAT-prior"),
        "ga_rand": ("tab:red", "-", "GA random pop"),
        "ga_vat": ("tab:red", "--", "GA VAT-seeded"),
    }
    for key, (c, ls, lab) in styles.items():
        curve, final, t, evals = primary[key]
        ax[0].plot(evals, curve, ls, color=c, label=f"{lab} ({t:.1f}s)")
    ax[0].axhline(
        primary["ref_single"], color="0.5", ls=":", lw=1, label="single VAT+2opt"
    )
    ax[0].axhline(
        primary["ref_multi"], color="k", ls=":", lw=1, label="best multi-start VAT+2opt"
    )
    ax[0].set_title(f"anytime convergence — {primary['name']} n={primary['n']}")
    ax[0].set_xlabel("tours evaluated (built + 2-opt'd)")
    ax[0].set_ylabel("% over optimum")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=7)
    ax[0].grid(True, which="both", alpha=0.3)

    # (b) final quality per method across instances
    ns = [s["n"] for s in summary]
    for key, (c, ls, lab) in styles.items():
        ax[1].plot(ns, [s[key][1] for s in summary], "o" + ls, color=c, label=lab)
    ax[1].plot(
        ns,
        [s["ref_multi"] for s in summary],
        "s:",
        color="k",
        label="best multi-start VAT+2opt",
    )
    ax[1].set_title("final quality vs n")
    ax[1].set_xlabel("n (cities)")
    ax[1].set_ylabel("% over optimum")
    ax[1].set_xscale("log")
    ax[1].legend(fontsize=7)
    ax[1].grid(True, which="both", alpha=0.3)

    # (c) hot-start delta: standard-init minus VAT-init final quality (>0 = prior helps)
    aco_delta = [s["aco_flat"][1] - s["aco_vat"][1] for s in summary]
    ga_delta = [s["ga_rand"][1] - s["ga_vat"][1] for s in summary]
    x = np.arange(len(summary))
    ax[2].bar(x - 0.18, aco_delta, 0.36, color="tab:blue", label="ACO: flat - VAT")
    ax[2].bar(x + 0.18, ga_delta, 0.36, color="tab:red", label="GA: rand - VAT")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{s['name']}\nn={s['n']}" for s in summary], fontsize=7)
    ax[2].set_title("hot-start benefit (final % pts; >0 = VAT prior better)")
    ax[2].set_ylabel("Δ % over optimum")
    ax[2].legend(fontsize=8)
    ax[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "VAT-consensus hot start for ACO & GA (memetic, soft prior)", fontsize=13
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_aco_ga.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if not gpu.is_available():
        raise SystemExit("no CUDA device — nothing to measure")
    print("VAT-consensus hot start for ACO & GA (memetic)")
    print("=" * 74)
    summary = []
    for tgt in (100, 200, 500):
        r = run_instance(tgt)
        summary.append(r)
        print(
            f"\n{r['name']} n={r['n']}  (ref: single VAT+2opt {r['ref_single']:.1f}%, "
            f"best multi-start {r['ref_multi']:.1f}%)"
        )
        for key, lab in (
            ("aco_flat", "ACO flat "),
            ("aco_vat", "ACO VAT  "),
            ("ga_rand", "GA random"),
            ("ga_vat", "GA VAT   "),
        ):
            _, final, t, _ = r[key]
            print(f"    {lab}: final {final:5.2f}%  ({t:5.1f}s)")
    primary = summary[1]  # n~200
    print(f"\nwrote {figure(primary, summary)}")
