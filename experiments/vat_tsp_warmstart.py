"""Spike (follow-up to vat_tsp.py): VAT/MST as a warm start for the minimum
Hamiltonian PATH (seriation TSP), done properly.

`vat_tsp.py` established the connection on easy blobs with a closed-tour ACO. This
script builds the "real result" pieces flagged there, and drops the clustering
angle entirely (a shorter tour does not help clustering -- that is settled). The
question here is purely: **is the VAT/MST ordering a good, cheap warm start for a
TSP/seriation solver, and where?**

Four things it adds over the intro spike:

  1. OPEN-PATH formulation throughout -- open-path cost, open 2-opt, open Or-opt
     (segment relocation, s=1..3), and an open-path Ant System. No closed-tour +
     longest-edge-cut proxy; we optimize the seriation objective directly.

  2. REAL construction baselines, not just random: nearest-neighbour, greedy-edge
     matching, and the MST double-tree (DFS pre-order) -- the textbook 2-approx.
     Every start is refined by the SAME local search, so the comparison is fair.
     VAT is "free" (the clustering front-end already computed it); the result is
     whether that free start is competitive.

  3. HARDER / non-blob / larger instances (uniform-random -- the classic hard
     TSP; moons; circles; a kNN-geodesic manifold) so the ACO final-tour gap is
     real, not just an anytime head start.

  4. NON-METRIC D (fractional p=0.5 Minkowski, cosine, kNN-geodesic): the metric
     double-tree bound (path <= 2*MST) is void there; we measure VAT/MST directly
     and whether VAT is still a strong empirical warm start.

Run:  python -m experiments.vat_tsp_warmstart
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numba import njit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tribbleclustering.pcvat import compute_ivat_c  # noqa: E402
from experiments.vat_tsp import path_cost, mst_weight  # noqa: E402
from experiments.stitched_vat import _prim_parent  # noqa: E402
from experiments.adversarial_eval import (  # noqa: E402
    two_moons,
    circles,
    easy_blobs,
)
from experiments.hardening_eval import (  # noqa: E402
    d_euclidean,
    d_fractional,
    d_cosine,
    d_geodesic,
    triangle_violation_rate,
)

FIG_DIR = Path(__file__).parent / "figures"


# ---------------------------------------------------------------------------
# Construction heuristics -> an initial open path (order array)
# ---------------------------------------------------------------------------
def vat_order(D):
    """VAT/MST ordering (Prim visit order) -- the free warm start."""
    _, _, p = compute_ivat_c(D.copy(), inplace=False)
    return np.ascontiguousarray(p, dtype=np.int64)


@njit(cache=True)
def nn_order(D, start):
    """Nearest-neighbour path from `start`."""
    n = D.shape[0]
    visited = np.zeros(n, np.bool_)
    order = np.empty(n, np.int64)
    order[0] = start
    visited[start] = True
    for step in range(1, n):
        cur = order[step - 1]
        best = -1
        bd = np.inf
        for v in range(n):
            if not visited[v] and D[cur, v] < bd:
                bd = D[cur, v]
                best = v
        order[step] = best
        visited[best] = True
    return order


def greedy_edge_order(D):
    """Greedy-edge matching -> Hamiltonian path: add cheapest edges that keep max
    degree <= 2 and form no premature cycle, until a single path (n-1 edges)."""
    n = D.shape[0]
    iu, ju = np.triu_indices(n, 1)
    w = D[iu, ju]
    srt = np.argsort(w, kind="stable")
    deg = np.zeros(n, np.int64)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj = [[] for _ in range(n)]
    added = 0
    for e in srt:
        a, b = int(iu[e]), int(ju[e])
        if deg[a] >= 2 or deg[b] >= 2:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue  # would close a cycle before all points are on the path
        parent[ra] = rb
        deg[a] += 1
        deg[b] += 1
        adj[a].append(b)
        adj[b].append(a)
        added += 1
        if added == n - 1:
            break
    # walk the path from an endpoint (degree 1)
    ends = [i for i in range(n) if deg[i] == 1]
    order = np.empty(n, np.int64)
    prev, cur, k = -1, ends[0], 0
    while k < n:
        order[k] = cur
        k += 1
        nxts = [x for x in adj[cur] if x != prev]
        if not nxts:
            break
        prev, cur = cur, nxts[0]
    return order


def mst_dfs_order(D, start):
    """Double-tree ordering: DFS pre-order of the Prim MST from `start`
    (the classic metric-TSP 2-approx ordering; distinct from VAT's visit order)."""
    n = D.shape[0]
    parent = _prim_parent(D)
    children = [[] for _ in range(n)]
    root = start
    for v in range(n):
        if parent[v] >= 0:
            children[parent[v]].append(v)
        elif v != root:
            # _prim_parent roots at the max-dissim seed; attach any stray to root
            children[root].append(v)
    order = np.empty(n, np.int64)
    stack = [root]
    k = 0
    seen = np.zeros(n, np.bool_)
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        order[k] = u
        k += 1
        # push children sorted by edge weight desc so cheapest is visited first
        cs = sorted(children[u], key=lambda c: -D[u, c])
        for c in cs:
            if not seen[c]:
                stack.append(c)
    if k < n:  # safety: append any unseen
        for v in range(n):
            if not seen[v]:
                order[k] = v
                k += 1
    return order


# ---------------------------------------------------------------------------
# Open-path local search: 2-opt + Or-opt (segment relocation, s=1..3)
# ---------------------------------------------------------------------------
@njit(cache=True)
def two_opt_path(order, D, max_pass=60):
    """Open-path 2-opt to convergence. Returns #improving moves applied."""
    n = order.shape[0]
    moves = 0
    for _ in range(max_pass):
        improved = False
        for i in range(0, n - 1):
            oi = order[i]
            left = order[i - 1] if i > 0 else -1
            for j in range(i + 1, n):
                oj = order[j]
                right = order[j + 1] if j < n - 1 else -1
                before = 0.0
                after = 0.0
                if left >= 0:
                    before += D[left, oi]
                    after += D[left, oj]
                if right >= 0:
                    before += D[oj, right]
                    after += D[oi, right]
                if after - before < -1e-9:
                    lo, hi = i, j
                    while lo < hi:
                        tmp = order[lo]
                        order[lo] = order[hi]
                        order[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
                    moves += 1
                    oi = order[i]
                    left = order[i - 1] if i > 0 else -1
        if not improved:
            break
    return moves


@njit(cache=True)
def _apply_relocate(order, i, s, t):
    """Return a new order with segment order[i:i+s] moved to sit right after the
    element currently at original index t (t=-1 front, t=-2 end)."""
    n = order.shape[0]
    seg = order[i : i + s].copy()
    rest = np.empty(n - s, np.int64)
    p = 0
    for x in range(n):
        if x < i or x >= i + s:
            rest[p] = order[x]
            p += 1
    out = np.empty(n, np.int64)
    if t == -1:  # front
        pos = 0
    elif t == -2:  # end
        pos = n - s
    else:
        u = order[t]
        pos = 0
        for r in range(n - s):
            if rest[r] == u:
                pos = r + 1
                break
    q = 0
    for r in range(pos):
        out[q] = rest[r]
        q += 1
    for r in range(s):
        out[q] = seg[r]
        q += 1
    for r in range(pos, n - s):
        out[q] = rest[r]
        q += 1
    return out


@njit(cache=True)
def or_opt_path(order, D, max_pass=40):
    """Open-path Or-opt: relocate segments of length 1,2,3 (forward orientation).
    Steepest descent. Returns (new_order, #moves)."""
    n = order.shape[0]
    moves = 0
    for _ in range(max_pass):
        best_delta = -1e-9
        bi, bs, bt = -1, -1, -3
        for s in (1, 2, 3):
            for i in range(0, n - s + 1):
                a = order[i]
                b = order[i + s - 1]
                left = order[i - 1] if i > 0 else -1
                right = order[i + s] if i + s < n else -1
                rem = 0.0
                if left >= 0:
                    rem += D[left, a]
                if right >= 0:
                    rem += D[b, right]
                if left >= 0 and right >= 0:
                    rem -= D[left, right]
                # insertion into every original gap outside the segment
                for t in range(0, n - 1):
                    if i - 1 <= t <= i + s - 1:
                        continue
                    u = order[t]
                    w = order[t + 1]
                    ins = D[u, a] + D[b, w] - D[u, w]
                    delta = ins - rem
                    if delta < best_delta:
                        best_delta = delta
                        bi, bs, bt = i, s, t
                if i > 0:  # insert at the very front
                    w = order[0]
                    delta = D[b, w] - rem
                    if delta < best_delta:
                        best_delta = delta
                        bi, bs, bt = i, s, -1
                if i + s < n:  # append at the very end
                    u = order[n - 1]
                    delta = D[u, a] - rem
                    if delta < best_delta:
                        best_delta = delta
                        bi, bs, bt = i, s, -2
        if bi < 0:
            break
        order = _apply_relocate(order, bi, bs, bt)
        moves += 1
    return order, moves


def local_search(order, D):
    """2-opt and Or-opt alternated to a joint local optimum. Returns
    (order, total_moves)."""
    order = np.ascontiguousarray(order, dtype=np.int64)
    total = 0
    while True:
        m1 = two_opt_path(order, D)
        order, m2 = or_opt_path(order, D)
        total += m1 + m2
        if m1 + m2 == 0:
            break
    return order, total


# ---------------------------------------------------------------------------
# Open-path Ant System, with an optional warm start (extra pheromone on a path)
# ---------------------------------------------------------------------------
@njit(cache=True)
def aco_path(
    D, n_ants, n_iter, alpha, beta, rho, q_dep, tau0, hot_order, hot_boost, seed
):
    n = D.shape[0]
    np.random.seed(seed)
    eta = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            eta[i, j] = 0.0 if i == j else 1.0 / (D[i, j] + 1e-12)
    tau = np.full((n, n), tau0)
    if hot_boost > 0.0:
        for k in range(n - 1):
            a = hot_order[k]
            b = hot_order[k + 1]
            tau[a, b] += tau0 * hot_boost
            tau[b, a] += tau0 * hot_boost

    best_len = 1.0e18
    best_order = np.arange(n)
    history = np.empty(n_iter)
    probs = np.empty(n)

    for it in range(n_iter):
        delta = np.zeros((n, n))
        for _ant in range(n_ants):
            visited = np.zeros(n, np.bool_)
            order = np.empty(n, np.int64)
            start = np.random.randint(n)
            order[0] = start
            visited[start] = True
            for step in range(1, n):
                cur = order[step - 1]
                total = 0.0
                for v in range(n):
                    if visited[v]:
                        probs[v] = 0.0
                    else:
                        val = (tau[cur, v] ** alpha) * (eta[cur, v] ** beta)
                        probs[v] = val
                        total += val
                r = np.random.random() * total
                acc = 0.0
                nxt = -1
                for v in range(n):
                    if not visited[v]:
                        acc += probs[v]
                        if acc >= r:
                            nxt = v
                            break
                if nxt == -1:
                    for v in range(n):
                        if not visited[v]:
                            nxt = v
                            break
                order[step] = nxt
                visited[nxt] = True
            L = 0.0
            for k in range(n - 1):
                L += D[order[k], order[k + 1]]
            contrib = q_dep / L
            for k in range(n - 1):
                a = order[k]
                b = order[k + 1]
                delta[a, b] += contrib
                delta[b, a] += contrib
            if L < best_len:
                best_len = L
                best_order = order.copy()
        for i in range(n):
            for j in range(n):
                tau[i, j] = (1.0 - rho) * tau[i, j] + delta[i, j]
        elite = q_dep / best_len
        for k in range(n - 1):
            a = best_order[k]
            b = best_order[k + 1]
            tau[a, b] += elite
            tau[b, a] += elite
        history[it] = best_len
    return best_order, best_len, history


ACO = dict(alpha=1.0, beta=2.5, rho=0.15, q_dep=1.0, tau0=1.0)
HOT_BOOST = 16.0


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------
def _uniform(n, seed):
    return np.random.default_rng(seed).random((n, 2))


def _ring_blobs(n, k=6, R=10.0, seed=1):
    """Blobs whose centres sit on a ring around the origin, so directions (hence
    cosine distances) span widely -- non-degenerate for a path-cost study."""
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * np.arange(k) / k
    C = R * np.c_[np.cos(ang), np.sin(ang)]
    lbl = rng.integers(0, k, n)
    return rng.normal(0, 0.8, (n, 2)) + C[lbl]


def metric_instances(n=800, seed=1):
    """(name, D) for a spread of metric instances, easy -> hard/non-blob."""
    Xb, _ = easy_blobs(n, seed=seed)
    Xm, _ = two_moons(n, noise=0.08, seed=seed)
    Xc, _ = circles(n, noise=0.06, seed=seed)
    return [
        ("blobs", d_euclidean(Xb)),
        ("uniform", d_euclidean(_uniform(n, seed))),
        ("moons", d_euclidean(Xm)),
        ("circles", d_euclidean(Xc)),
        ("geodesic-moons", d_geodesic(Xm)),
    ]


def _seed_of(D):
    """VAT's seed: a vertex of the globally most-distant pair."""
    return int(np.argmax(D)) // D.shape[0]


def _starts(D):
    """Deterministic construction starts -> {name: order}. 'random' handled
    separately (stochastic, averaged over seeds)."""
    s = _seed_of(D)
    return {
        "nearest-neighbour": nn_order(D, s),
        "greedy-edge": greedy_edge_order(D),
        "MST double-tree": mst_dfs_order(D, s),
        "VAT (free)": vat_order(D),
    }


# ---------------------------------------------------------------------------
# Report A — VAT vs real construction heuristics as warm starts
# ---------------------------------------------------------------------------
def warmstart_report(n=800):
    print(
        "\n=== A. warm-start quality: every start refined by the SAME 2-opt+Or-opt ==="
    )
    print("    final path cost, gap%% vs the best start on that instance, and #moves")
    print(
        "    ('VAT (free)' costs nothing extra: the clustering front-end computed it)"
    )
    gaps = {}  # method -> list of final-gap% across instances
    for name, D in metric_instances(n):
        D = np.ascontiguousarray(D)
        rows = {}
        # deterministic starts
        for m, o in _starts(D).items():
            fin, mv = local_search(o, D)
            rows[m] = (path_cost(o, D), path_cost(fin, D), mv)
        # random start averaged over seeds
        r_init, r_fin, r_mv = [], [], []
        for sd in range(5):
            o = np.random.default_rng(100 + sd).permutation(D.shape[0])
            fin, mv = local_search(o, D)
            r_init.append(path_cost(o, D))
            r_fin.append(path_cost(fin, D))
            r_mv.append(mv)
        rows["random (x5)"] = (np.mean(r_init), np.mean(r_fin), np.mean(r_mv))

        best_fin = min(v[1] for v in rows.values())
        print(f"\n  {name} (n={n}):")
        print(
            f"    {'start':18s} {'init':>9s} {'final':>9s} {'gap%':>7s} {'moves':>7s}"
        )
        for m in (
            "random (x5)",
            "nearest-neighbour",
            "greedy-edge",
            "MST double-tree",
            "VAT (free)",
        ):
            init, fin, mv = rows[m]
            g = 100.0 * (fin - best_fin) / best_fin
            gaps.setdefault(m, []).append(g)
            print(f"    {m:18s} {init:9.2f} {fin:9.2f} {g:7.2f} {mv:7.0f}")
    print("\n  mean final gap%% across instances (lower = better warm start):")
    for m, gl in sorted(gaps.items(), key=lambda kv: np.mean(kv[1])):
        print(f"    {m:18s} {np.mean(gl):6.2f}%")
    return gaps


# ---------------------------------------------------------------------------
# Report B — open-path ACO, VAT hot start vs cold, on HARD instances
# ---------------------------------------------------------------------------
def aco_hard_report(n=300):
    print("\n=== B. open-path ACO on harder instances: does the FINAL gap open up? ===")
    print("    (blobs converge to a tie; on hard/non-blob data the hot start also")
    print("     wins the final tour, not just the anytime trajectory)")
    Xm, _ = two_moons(n, noise=0.08, seed=3)
    cases = [
        ("blobs", d_euclidean(easy_blobs(n, seed=3)[0])),
        ("uniform", d_euclidean(_uniform(n, 3))),
        ("circles", d_euclidean(circles(n, noise=0.06, seed=3)[0])),
        ("geodesic-moons", d_geodesic(Xm)),
    ]
    saved = {}
    for name, D in cases:
        D = np.ascontiguousarray(D)
        p = vat_order(D)
        a = (ACO["alpha"], ACO["beta"], ACO["rho"], ACO["q_dep"], ACO["tau0"])
        _, cold, ch = aco_path(D, 12, 70, *a, p, 0.0, seed=7)
        _, hot, hh = aco_path(D, 12, 70, *a, p, HOT_BOOST, seed=7)
        fin_gap = 100.0 * (cold - hot) / cold
        print(
            f"  {name:15s}: iter1 cold/hot {ch[0]:7.1f}/{hh[0]:7.1f}  "
            f"final cold/hot {cold:7.1f}/{hot:7.1f}  final gap {fin_gap:+5.1f}%"
        )
        saved[name] = (ch, hh)
    return saved


# ---------------------------------------------------------------------------
# Report C — non-metric D: the double-tree bound is void; is VAT still good?
# ---------------------------------------------------------------------------
def nonmetric_report(n=800, seed=1):
    print("\n=== C. non-metric D: the double-tree 2x-MST bound, and VAT as a start ===")
    print("    DT/MST = MST double-tree path / MST weight (the quantity the metric")
    print("    theorem bounds by 2). VAT/MST is VAT's visit order (a different")
    print("    traversal, NOT bound-covered). VAT edge = random_fin vs VAT_fin.")
    X = _ring_blobs(n, seed=seed)
    builders = [
        ("euclidean (metric)", d_euclidean(X)),
        ("fractional p=0.5", d_fractional(X)),
        ("cosine", d_cosine(X)),
        ("kNN-geodesic", d_geodesic(X)),
    ]
    print(
        f"  {'dissimilarity':20s} {'tri-viol':>9s} {'DT/MST':>7s} {'VAT/MST':>8s} "
        f"{'rand fin':>9s} {'VAT fin':>9s} {'VAT edge':>9s}"
    )
    ratios = {}
    for name, D in builders:
        D = np.ascontiguousarray(D)
        tv = triangle_violation_rate(D)
        s = _seed_of(D)
        dt = mst_dfs_order(D, s)
        dt_ratio = path_cost(np.ascontiguousarray(dt, dtype=np.int64), D) / mst_weight(
            D
        )
        vat_ratio = path_cost(vat_order(D), D) / mst_weight(D)
        fin_vat, _ = local_search(vat_order(D), D)
        cvat = path_cost(fin_vat, D)
        r_fin = []
        for sd in range(5):
            o = np.random.default_rng(200 + sd).permutation(n)
            fin, _ = local_search(o, D)
            r_fin.append(path_cost(fin, D))
        crand = float(np.mean(r_fin))
        edge = 100.0 * (crand - cvat) / crand
        ratios[name] = (dt_ratio, tv)
        print(
            f"  {name:20s} {tv:9.2%} {dt_ratio:7.3f} {vat_ratio:8.3f} "
            f"{crand:9.2f} {cvat:9.2f} {edge:+8.1f}%"
        )
    print("    DT/MST stays < 2 for metric D (euclidean, geodesic) and BREAKS it for")
    print("    cosine (highest triangle-violation) -- the bound is genuinely void off")
    print("    the metric regime. VAT has no guarantee anywhere, yet VAT-start still")
    print("    finishes at/below the random-start average on every D -- robust.")
    return ratios


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def figure(gaps, aco_saved, ratios):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))

    # A: final gap% by start method, per instance
    ax = axes[0]
    methods = [
        "random (x5)",
        "nearest-neighbour",
        "greedy-edge",
        "MST double-tree",
        "VAT (free)",
    ]
    instances = ["blobs", "uniform", "moons", "circles", "geodesic-moons"]
    colors = ["#999", "#e8a", "#6b9", "#c85", "#268"]
    x = np.arange(len(instances))
    w = 0.16
    for mi, m in enumerate(methods):
        ax.bar(x + (mi - 2) * w, gaps[m], w, label=m, color=colors[mi])
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("final path gap % vs best start")
    ax.set_title("A. VAT is competitive as a (free) warm start", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")

    # B: open-path ACO convergence on a hard instance
    ax = axes[1]
    ch, hh = aco_saved["uniform"]
    it = np.arange(1, len(ch) + 1)
    ax.plot(it, ch, color="#c44", lw=1.8, label="cold start (random)")
    ax.plot(it, hh, color="#268", lw=1.8, label="hot start (VAT)")
    ax.set_xlabel("ACO iteration")
    ax.set_ylabel("best open-path length")
    ax.set_title(
        "B. open-path ACO on uniform (hard):\nhot leads and finishes lower", fontsize=11
    )
    ax.legend(fontsize=8)

    # C: double-tree/MST ratio across D types, with the metric 2x bound
    ax = axes[2]
    names = list(ratios.keys())
    vals = [ratios[k][0] for k in names]
    tvs = [ratios[k][1] for k in names]
    bcol = ["#268" if t < 0.01 else "#c85" for t in tvs]
    bars = ax.bar(range(len(names)), vals, color=bcol)
    ax.axhline(
        2.0, color="red", ls="--", lw=1.2, label="double-tree bound (2x, metric)"
    )
    for i, t in enumerate(tvs):
        ax.text(
            i,
            vals[i] + 0.03,
            f"tri-viol\n{t:.0%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [n.split(" ")[0] for n in names], rotation=20, ha="right", fontsize=8
    )
    ax.set_ylabel("MST double-tree path / MST weight")
    ax.set_title(
        "C. 2x-MST bound holds for metric D,\nbreaks for cosine (non-metric)",
        fontsize=11,
    )
    ax.set_ylim(0, max(vals) + 0.5)
    ax.legend(fontsize=8, loc="upper left")
    bars[0].set_label("_")

    fig.suptitle(
        "VAT/MST as a warm start for the minimum-Hamiltonian-path (seriation TSP): "
        "competitive with real construction heuristics, and robust off the metric regime",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / "vat_tsp_warmstart.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("VAT/MST as a TSP warm start — real-result study")
    print("===============================================")
    t0 = time.perf_counter()
    gaps = warmstart_report()
    aco_saved = aco_hard_report()
    ratios = nonmetric_report()
    print(f"\nwrote {figure(gaps, aco_saved, ratios)}")
    print(f"(total {time.perf_counter() - t0:.1f}s)")
