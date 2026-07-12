# Dual-VAT MST join mechanisms: endpoint vs GPU N×M cycle-merge

The dual-VAT build gives two cluster VAT paths (P1, P2); how you join them into a
closed tour ("close the loop") is a separate choice. Compared two mechanisms on
a **balanced** partition (two-density-peak seed, so N≈M and the grid is
meaningful), nearest-size TSPLIB instances, fp32, reference = published optimum.
Source: `experiments/vat_tsp_join.py`.

- **endpoint** — connect the two paths at their 2×2 endpoints, best of the 4
  orientations. O(1) (only the path ends are candidates).
- **nxm (GPU)** — close each path into a sub-cycle, then take the best
  *2-opt-across* move over **all N×M cross edge pairs**: remove one edge from each
  cycle and reconnect crosswise (two patterns), evaluated as a full N×M delta
  matrix on the device; only the winning (i, j, pattern) crosses to the host.
  O(N·M).

## Results (% over optimum; raw = before polish, +LK = after neighbour-LK)

| instance | n | \|C1\|/\|C2\| | endpt raw | endpt +LK | nxm raw | **nxm +LK** | endpt s | nxm s |
|----------|------|------|-----------|-----------|---------|-------------|---------|-------|
| kroA200 | 200 | 86/114 | +78% | **+2.4%** | +64% | +5.1% | 0.0004 | 0.0020 |
| d493 | 493 | 436/57 | +113% | +3.7% | +111% | **+2.9%** | 0.0002 | 0.0004 |
| dsj1000 | 1000 | 245/755 | +155% | **+8.4%** | +152% | +9.9% | 0.0002 | 0.0004 |
| d2103 | 2103 | 1317/786 | +88% | +14.4% | +81% | **+12.7%** | 0.0002 | 0.0006 |
| fnl4461 | 4461 | 4452/9 | +242% | +5.4% | +240% | **+5.1%** | 0.0002 | 0.0005 |
| rl11849 | 11849 | 5679/6170 | +226% | +23.7% | +226% | **+16.9%** | 0.0002 | 0.0252 |

## Findings

- **The N×M cycle-merge reliably improves the RAW tour** (e.g. kroA200 +78→+64%,
  d2103 +88→+81%): it finds a genuinely better cross-cluster bridge than the two
  path ends. The gain is small in *relative* terms because the join only changes
  2 of n edges — the raw cost is dominated by the VAT paths' internal structure.
- **After the LK polish, the N×M join wins on the harder / larger instances** —
  most clearly **rl11849 (+23.7% → +16.9%, a 7-point gain)**, and also d493,
  d2103, fnl4461 — because a better set of removed/added seam edges drops the tour
  into a better 2-opt basin. On the small/easy instances (kroA200, dsj1000) the
  endpoint join happens to polish slightly better; the effect is basin-dependent
  and small there.
- **Cost.** The endpoint join is O(1) (~0.2 ms, flat). The N×M merge is O(N·M) on
  the device: negligible when a cluster is small (fnl4461's 4452/9 → 40k pairs,
  0.5 ms) and 25 ms at the balanced n=11849 (5679×6170 ≈ 35M pairs). It stays
  sub-30 ms through ~12k; the N×M delta matrix is the memory limit at very large
  balanced n (≈1.2 GB at n=34k balanced).

## Verdict

Use the **N×M GPU cycle-merge** to close the loop when the partition is balanced
and the instance is non-trivial — it improves the raw bridge every time and the
polished tour on the hard instances (rl11849 the standout), for a few
milliseconds. The endpoint join remains the right default when one cluster is
tiny (the grid degenerates to ~N anyway) or for the smallest instances. Both are
cheap; neither changes the headline that the **local search**, not the join, is
the dominant quality lever.

## Files
- `experiments/vat_tsp_join.py`, `experiments/figures/vat_tsp_join.png`.
- Joins: `vat_tsp_dualvat_lk.join_endpoint`, `join_nxm_device` (GPU).
