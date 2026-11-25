//! Rust port of `tribbleclustering.cdlmst` (certificate-and-filter MST for
//! VAT/IVAT) plus a compact dense Prim, for a language-level comparison in
//! `benchmarks/mst_pareto.py`. Same algorithm, same passes, same tie order as
//! the Cython kernels; see `src/tribbleclustering/dlmst.py` for the design.
//!
//! `d == 0` means `data` is an n x n distance matrix; otherwise it is n x d
//! points and L2 distances are computed on the fly (O(n) extra memory).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AO};
use std::sync::Mutex;

pub trait F:
    Copy
    + PartialOrd
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + 'static
{
    const INF: Self;
    const NINF: Self;
    const ZERO: Self;
    fn sqrt(self) -> Self;
    fn total(self, o: Self) -> Ordering;
}
impl F for f64 {
    const INF: Self = f64::INFINITY;
    const NINF: Self = f64::NEG_INFINITY;
    const ZERO: Self = 0.0;
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn total(self, o: Self) -> Ordering {
        self.total_cmp(&o)
    }
}
impl F for f32 {
    const INF: Self = f32::INFINITY;
    const NINF: Self = f32::NEG_INFINITY;
    const ZERO: Self = 0.0;
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn total(self, o: Self) -> Ordering {
        self.total_cmp(&o)
    }
}

/// Raw pointer that may be shared across threads; every use writes disjoint
/// indices.
#[derive(Clone, Copy)]
struct P<T>(*mut T);
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}
impl<T> P<T> {
    #[inline(always)]
    unsafe fn w(self, i: usize, v: T) {
        *self.0.add(i) = v;
    }
}

const PAR_THRESHOLD: usize = 512;
const CHUNK: usize = 64;

struct Src<'a, T: F> {
    r: Option<&'a [T]>,
    x: &'a [T],
    n: usize,
    d: usize,
}

impl<'a, T: F> Src<'a, T> {
    #[inline(always)]
    fn dist(&self, i: usize, j: usize) -> T {
        if let Some(r) = self.r {
            return r[i * self.n + j];
        }
        let d = self.d;
        let x = self.x.as_ptr();
        let mut s = T::ZERO;
        // SAFETY: i, j < n and x has n * d elements.
        unsafe {
            let (a, b) = (x.add(i * d), x.add(j * d));
            for k in 0..d {
                let t = *a.add(k) - *b.add(k);
                s = s + t * t;
            }
        }
        s.sqrt()
    }
    #[inline(always)]
    fn row<'b>(&'b self, i: usize, buf: &'b mut Vec<T>) -> &'b [T] {
        if let Some(r) = self.r {
            return &r[i * self.n..(i + 1) * self.n];
        }
        for j in 0..self.n {
            buf[j] = self.dist(i, j);
        }
        &buf[..]
    }
}

/// Run `f(thread_id, row)` over rows `0..n` with dynamic chunking.
fn par_rows<Fn_: Fn(usize, usize) + Sync>(n: usize, nt: usize, chunk: usize, f: Fn_) {
    if nt <= 1 {
        for i in 0..n {
            f(0, i);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for t in 0..nt {
            let (next, f) = (&next, &f);
            s.spawn(move || loop {
                let a = next.fetch_add(chunk, AO::Relaxed);
                if a >= n {
                    break;
                }
                for i in a..(a + chunk).min(n) {
                    f(t, i);
                }
            });
        }
    });
}

fn find(par: &mut [u32], mut x: u32) -> u32 {
    let mut r = x;
    while par[r as usize] != r {
        r = par[r as usize];
    }
    while par[x as usize] != r {
        let nx = par[x as usize];
        par[x as usize] = r;
        x = nx;
    }
    r
}

/// Kruskal; returns indices of chosen edges. Tie order (w, max, min) as in
/// dlmst.py / cdlmst.pyx.
fn kruskal<T: F>(n: usize, u: &[u32], v: &[u32], w: &[T]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..u.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        w[a].total(w[b])
            .then(u[a].max(v[a]).cmp(&u[b].max(v[b])))
            .then(u[a].min(v[a]).cmp(&u[b].min(v[b])))
            .then(a.cmp(&b))
    });
    let mut par: Vec<u32> = (0..n as u32).collect();
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    for e in idx {
        let (a, b) = (find(&mut par, u[e]), find(&mut par, v[e]));
        if a != b {
            par[a as usize] = b;
            out.push(e);
            if out.len() + 1 == n {
                break;
            }
        }
    }
    out
}

/// Leaf order of the Kruskal reconstruction tree of spanning tree (u, v, w)
/// and g[t] = merge weight between order[t] and order[t + 1].
fn krt<T: F>(n: usize, u: &[u32], v: &[u32], w: &[T]) -> (Vec<u32>, Vec<T>) {
    let mut srt: Vec<usize> = (0..u.len()).collect();
    srt.sort_by(|&a, &b| w[a].total(w[b]));
    let mut par: Vec<u32> = (0..n as u32).collect();
    let mut head: Vec<u32> = (0..n as u32).collect();
    let tail: Vec<u32> = (0..n as u32).collect();
    let mut nxt = vec![u32::MAX; n];
    let mut gw = vec![T::ZERO; n];
    for e in srt {
        let (a, b) = (find(&mut par, u[e]), find(&mut par, v[e]));
        nxt[tail[a as usize] as usize] = head[b as usize];
        gw[tail[a as usize] as usize] = w[e];
        head[b as usize] = head[a as usize];
        par[a as usize] = b;
    }
    let mut x = head[find(&mut par, 0) as usize];
    let mut order = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n.saturating_sub(1));
    for t in 0..n {
        order.push(x);
        if t + 1 < n {
            g.push(gw[x as usize]);
        }
        x = nxt[x as usize];
    }
    (order, g)
}

struct Mst<T> {
    u: Vec<u32>,
    v: Vec<u32>,
    w: Vec<T>,
    seed_i: u32,
    seed_j: u32,
}

fn dlmst<T: F>(s: &Src<T>, k: usize, nt: usize) -> Mst<T> {
    let n = s.n;
    let k = k.clamp(1, n - 1);
    // Pass 1: k lightest per row + VAT seed.
    let mut ev = vec![u32::MAX; n * k];
    let mut ew = vec![T::INF; n * k];
    let mut rmax = vec![T::NINF; n];
    let mut rarg = vec![0u32; n];
    {
        let (pev, pew, prm, pra) = (P(ev.as_mut_ptr()), P(ew.as_mut_ptr()), P(rmax.as_mut_ptr()), P(rarg.as_mut_ptr()));
        let bufs: Vec<Mutex<Vec<T>>> = (0..nt).map(|_| Mutex::new(if s.r.is_none() { vec![T::ZERO; n] } else { Vec::new() })).collect();
        par_rows(n, nt, 16, |t, i| {
            let mut bg = bufs[t].lock().unwrap();
            let row = s.row(i, &mut bg);
            let mut kw = vec![T::INF; k];
            let mut kv = vec![u32::MAX; k];
            let (mut mx, mut am) = (T::NINF, 0usize);
            let mut j0 = 0;
            while j0 < n {
                let j1 = (j0 + CHUNK).min(n);
                let (mut cmin, mut cmax) = (row[j0], row[j0]);
                for &x in &row[j0 + 1..j1] {
                    cmin = if x < cmin { x } else { cmin };
                    cmax = if x > cmax { x } else { cmax };
                }
                if cmax > mx {
                    for j in j0..j1 {
                        if row[j] > mx {
                            mx = row[j];
                            am = j;
                        }
                    }
                }
                if cmin < kw[k - 1] || (j0 <= i && i < j1) {
                    for j in j0..j1 {
                        let x = row[j];
                        if j == i || !(x < kw[k - 1]) {
                            continue;
                        }
                        let mut t = k - 1;
                        while t > 0 && kw[t - 1] > x {
                            kw[t] = kw[t - 1];
                            kv[t] = kv[t - 1];
                            t -= 1;
                        }
                        kw[t] = x;
                        kv[t] = j as u32;
                    }
                }
                j0 = j1;
            }
            unsafe {
                for t in 0..k {
                    pev.w(i * k + t, kv[t]);
                    pew.w(i * k + t, kw[t]);
                }
                prm.w(i, mx);
                pra.w(i, am as u32);
            }
        });
    }
    let mut seed_i = 0;
    for i in 1..n {
        if rmax[i] > rmax[seed_i] {
            seed_i = i;
        }
    }
    let seed_j = rarg[seed_i];
    let mut eu = Vec::with_capacity(n * k);
    let (mut ev2, mut ew2) = (Vec::with_capacity(n * k), Vec::with_capacity(n * k));
    for i in 0..n {
        for t in 0..k {
            if ev[i * k + t] != u32::MAX {
                eu.push(i as u32);
                ev2.push(ev[i * k + t]);
                ew2.push(ew[i * k + t]);
            }
        }
    }
    let keep = kruskal(n, &eu, &ev2, &ew2);
    let mut fu: Vec<u32> = keep.iter().map(|&e| eu[e]).collect();
    let mut fv: Vec<u32> = keep.iter().map(|&e| ev2[e]).collect();
    let mut fw: Vec<T> = keep.iter().map(|&e| ew2[e]).collect();

    // Pass 2: contraction (or Borůvka merges) until F spans.
    while fu.len() + 1 < n {
        let mut par: Vec<u32> = (0..n as u32).collect();
        for e in 0..fu.len() {
            let (a, b) = (find(&mut par, fu[e]), find(&mut par, fv[e]));
            if a != b {
                par[a as usize] = b;
            }
        }
        let mut id = vec![u32::MAX; n];
        let mut comp = vec![0u32; n];
        let mut c = 0u32;
        for i in 0..n {
            let r = find(&mut par, i as u32) as usize;
            if id[r] == u32::MAX {
                id[r] = c;
                c += 1;
            }
            comp[i] = id[r];
        }
        let c = c as usize;
        let (mu, mv, mw): (Vec<u32>, Vec<u32>, Vec<T>) = if c <= 256 {
            let cc = c * c;
            let tabs: Vec<Mutex<(Vec<T>, Vec<u32>, Vec<u32>)>> = (0..nt)
                .map(|_| {
                    let mut tw = vec![T::INF; cc];
                    for a in 0..c {
                        tw[a * c + a] = T::NINF;
                    }
                    Mutex::new((tw, vec![u32::MAX; cc], vec![u32::MAX; cc]))
                })
                .collect();
            par_rows(n.saturating_sub(1), nt, 16, |t, i| {
                let mut g = tabs[t].lock().unwrap();
                let (tw, tu, tv) = &mut *g;
                let ci = comp[i] as usize;
                let base = ci * c;
                for j in i + 1..n {
                    let cj = comp[j] as usize;
                    if s.r.is_none() && cj == ci {
                        continue;
                    }
                    let x = s.dist(i, j);
                    if x < tw[base + cj] {
                        tw[base + cj] = x;
                        tu[base + cj] = i as u32;
                        tv[base + cj] = j as u32;
                    }
                }
            });
            let (mut bw, mut bu, mut bv) = (vec![T::INF; cc], vec![u32::MAX; cc], vec![u32::MAX; cc]);
            for m in tabs {
                let (tw, tu, tv) = m.into_inner().unwrap();
                for t in 0..cc {
                    if tu[t] != u32::MAX && tw[t] < bw[t] {
                        bw[t] = tw[t];
                        bu[t] = tu[t];
                        bv[t] = tv[t];
                    }
                }
            }
            let mut o = (Vec::new(), Vec::new(), Vec::new());
            for t in 0..cc {
                if bu[t] != u32::MAX {
                    o.0.push(bu[t]);
                    o.1.push(bv[t]);
                    o.2.push(bw[t]);
                }
            }
            o
        } else {
            let mut bv = vec![0u32; n];
            let mut bw = vec![T::INF; n];
            let (pbv, pbw) = (P(bv.as_mut_ptr()), P(bw.as_mut_ptr()));
            par_rows(n, nt, 16, |_, i| {
                let (mut best, mut bj) = (T::INF, 0u32);
                for j in 0..n {
                    if comp[j] != comp[i] {
                        let x = s.dist(i, j);
                        if x < best {
                            best = x;
                            bj = j as u32;
                        }
                    }
                }
                unsafe {
                    pbv.w(i, bj);
                    pbw.w(i, best);
                }
            });
            ((0..n as u32).collect(), bv, bw)
        };
        let cu: Vec<u32> = fu.iter().chain(mu.iter()).copied().collect();
        let cv: Vec<u32> = fv.iter().chain(mv.iter()).copied().collect();
        let cw: Vec<T> = fw.iter().chain(mw.iter()).copied().collect();
        let keep = kruskal(n, &cu, &cv, &cw);
        fu = keep.iter().map(|&e| cu[e]).collect();
        fv = keep.iter().map(|&e| cv[e]).collect();
        fw = keep.iter().map(|&e| cw[e]).collect();
    }

    // Pass 3: filter. Row i in KRT leaf order: minimax = running max of g.
    let (order, g) = krt(n, &fu, &fv, &fw);
    let xp: Vec<T> = if s.r.is_none() {
        let d = s.d;
        let mut v = Vec::with_capacity(n * d);
        for &o in &order {
            v.extend_from_slice(&s.x[o as usize * d..(o as usize + 1) * d]);
        }
        v
    } else {
        Vec::new()
    };
    let sp = Src { r: None, x: &xp, n, d: s.d };
    let found: Vec<Mutex<Vec<(u32, u32, T)>>> = (0..nt).map(|_| Mutex::new(Vec::new())).collect();
    par_rows(n.saturating_sub(1), nt, 16, |t, i| {
        let oi = order[i] as usize;
        let mut mm = T::NINF;
        let mut loc: Vec<(u32, u32, T)> = Vec::new();
        if let Some(r) = s.r {
            let row = &r[oi * n..(oi + 1) * n];
            for j in i + 1..n {
                if g[j - 1] > mm {
                    mm = g[j - 1];
                }
                // SAFETY: order holds vertex ids < n.
                let x = unsafe { *row.get_unchecked(*order.get_unchecked(j) as usize) };
                if x < mm {
                    loc.push((oi as u32, order[j], x));
                }
            }
        } else {
            for j in i + 1..n {
                if g[j - 1] > mm {
                    mm = g[j - 1];
                }
                let x = sp.dist(i, j);
                if x < mm {
                    loc.push((oi as u32, order[j], x));
                }
            }
        }
        if !loc.is_empty() {
            found[t].lock().unwrap().extend(loc);
        }
    });
    for m in found {
        for (a, b, x) in m.into_inner().unwrap() {
            fu.push(a);
            fv.push(b);
            fw.push(x);
        }
    }
    let keep = kruskal(n, &fu, &fv, &fw);
    Mst {
        u: keep.iter().map(|&e| fu[e]).collect(),
        v: keep.iter().map(|&e| fv[e]).collect(),
        w: keep.iter().map(|&e| fw[e]).collect(),
        seed_i: seed_i as u32,
        seed_j,
    }
}

struct HeapItem<T: F>(T, u32, i64);
impl<T: F> PartialEq for HeapItem<T> {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl<T: F> Eq for HeapItem<T> {}
impl<T: F> PartialOrd for HeapItem<T> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<T: F> Ord for HeapItem<T> {
    // Reversed: BinaryHeap is a max-heap.
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total(self.0).then(o.1.cmp(&self.1))
    }
}

/// Prim's visiting order on the MST: (p, q 1-based parent position, w).
fn tree_prim<T: F>(n: usize, m: &Mst<T>, p: &mut [i32], q: &mut [i32], w: &mut [T]) {
    let mut deg = vec![0usize; n + 1];
    for e in 0..m.u.len() {
        deg[m.u[e] as usize + 1] += 1;
        deg[m.v[e] as usize + 1] += 1;
    }
    for i in 0..n {
        deg[i + 1] += deg[i];
    }
    let mut cur = deg.clone();
    let mut adj = vec![(0u32, T::ZERO); 2 * m.u.len()];
    for e in 0..m.u.len() {
        let (a, b) = (m.u[e] as usize, m.v[e] as usize);
        adj[cur[a]] = (b as u32, m.w[e]);
        cur[a] += 1;
        adj[cur[b]] = (a as u32, m.w[e]);
        cur[b] += 1;
    }
    let mut pos = vec![-1i64; n];
    let mut h = BinaryHeap::with_capacity(n);
    h.push(HeapItem(T::ZERO, m.seed_i, -1));
    let mut r = 0usize;
    while let Some(HeapItem(wt, v, par)) = h.pop() {
        if pos[v as usize] >= 0 {
            continue;
        }
        pos[v as usize] = r as i64;
        p[r] = v as i32;
        q[r] = if par < 0 { m.seed_j as i32 } else { pos[par as usize] as i32 + 1 };
        w[r] = wt;
        r += 1;
        for &(x, xw) in &adj[deg[v as usize]..deg[v as usize + 1]] {
            if pos[x as usize] < 0 {
                h.push(HeapItem(xw, x, v as i64));
            }
        }
    }
}

fn nthreads(n: usize, req: i32) -> usize {
    let avail = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let nt = if req > 0 { req as usize } else { avail };
    if n < PAR_THRESHOLD { 1 } else { nt }
}

unsafe fn order_impl<T: F>(data: *const T, n: usize, d: usize, k: usize, nt: i32, p: *mut i32, q: *mut i32, w: *mut T) -> i32 {
    if n < 2 {
        *p = 0;
        *q = 0;
        return 0;
    }
    let len = if d == 0 { n * n } else { n * d };
    let sl = std::slice::from_raw_parts(data, len);
    let s = if d == 0 { Src { r: Some(sl), x: &[], n, d: 0 } } else { Src { r: None, x: sl, n, d } };
    let m = dlmst(&s, k, nthreads(n, nt));
    tree_prim(n, &m, std::slice::from_raw_parts_mut(p, n), std::slice::from_raw_parts_mut(q, n), std::slice::from_raw_parts_mut(w, n));
    0
}

/// IVAT from the MST: out (n x n) and p (n).
unsafe fn ivat_impl<T: F>(data: *const T, n: usize, d: usize, k: usize, nt: i32, out: *mut T, p: *mut i32) -> i32 {
    if n < 2 {
        *out = T::ZERO;
        *p = 0;
        return 0;
    }
    let len = if d == 0 { n * n } else { n * d };
    let sl = std::slice::from_raw_parts(data, len);
    let s = if d == 0 { Src { r: Some(sl), x: &[], n, d: 0 } } else { Src { r: None, x: sl, n, d } };
    let nt = nthreads(n, nt);
    let m = dlmst(&s, k, nt);
    let ps = std::slice::from_raw_parts_mut(p, n);
    let mut q = vec![0i32; n];
    let mut w = vec![T::ZERO; n];
    tree_prim(n, &m, ps, &mut q, &mut w);
    let (order, g) = krt(n, &m.u, &m.v, &m.w);
    let mut vpos = vec![0u32; n];
    for (r, &v) in ps.iter().enumerate() {
        vpos[v as usize] = r as u32;
    }
    let mut lpos = vec![0usize; n];
    let mut col = vec![0u32; n];
    for (j, &o) in order.iter().enumerate() {
        lpos[o as usize] = j;
        col[j] = vpos[o as usize];
    }
    let po = P(out);
    let ps: &[i32] = ps;
    par_rows(n, nt, 16, |_, r| {
        let base = r * n;
        let i = lpos[ps[r] as usize];
        po.w(base + r, T::ZERO);
        let mut mm = T::NINF;
        for j in i + 1..n {
            if g[j - 1] > mm {
                mm = g[j - 1];
            }
            po.w(base + col[j] as usize, mm);
        }
        mm = T::NINF;
        for j in (0..i).rev() {
            if g[j] > mm {
                mm = g[j];
            }
            po.w(base + col[j] as usize, mm);
        }
    });
    0
}

/// Compact-active-set dense Prim (the algorithm in pcvat.pyx), matrix only.
unsafe fn prim_impl<T: F>(data: *const T, n: usize, nt: i32, p: *mut i32, q: *mut i32) -> i32 {
    let r = std::slice::from_raw_parts(data, n * n);
    let p = std::slice::from_raw_parts_mut(p, n);
    let q = std::slice::from_raw_parts_mut(q, n);
    let nt = nthreads(n, nt);
    // Parallel global-max seed scan (row-major first max), as in pcvat.
    let mut rmax = vec![T::NINF; n];
    let mut rarg = vec![0u32; n];
    let (pm, pa) = (P(rmax.as_mut_ptr()), P(rarg.as_mut_ptr()));
    par_rows(n, nt, 64, |_, i| {
        let row = &r[i * n..(i + 1) * n];
        let (mut mx, mut am) = (T::NINF, 0u32);
        for (j, &x) in row.iter().enumerate() {
            if x > mx {
                mx = x;
                am = j as u32;
            }
        }
        pm.w(i, mx);
        pa.w(i, am);
    });
    let mut si = 0;
    for i in 1..n {
        if rmax[i] > rmax[si] {
            si = i;
        }
    }
    let mut av: Vec<u32> = (0..n as u32).collect();
    let mut ak = vec![T::INF; n];
    let mut ap = vec![-1i32; n];
    ak[si] = rmax[si];
    ap[si] = rarg[si] as i32;
    let mut bk = si;
    let mut m = n;
    for rnd in 0..n {
        let u = av[bk] as usize;
        p[rnd] = u as i32;
        q[rnd] = ap[bk];
        m -= 1;
        av[bk] = av[m];
        ak[bk] = ak[m];
        ap[bk] = ap[m];
        let row = &r[u * n..(u + 1) * n];
        let mut best = T::INF;
        bk = usize::MAX;
        let (avp, akp, app) = (av.as_ptr(), ak.as_mut_ptr(), ap.as_mut_ptr());
        // SAFETY: i < m <= n for all three arrays; av holds ids < n.
        for i in 0..m {
            let x = *row.get_unchecked(*avp.add(i) as usize);
            let ki = akp.add(i);
            if x < *ki {
                *ki = x;
                *app.add(i) = rnd as i32 + 1;
            }
            if *ki < best {
                best = *ki;
                bk = i;
            }
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn dlmst_order_f64(data: *const f64, n: usize, d: usize, k: usize, nt: i32, p: *mut i32, q: *mut i32, w: *mut f64) -> i32 {
    order_impl(data, n, d, k, nt, p, q, w)
}
#[no_mangle]
pub unsafe extern "C" fn dlmst_order_f32(data: *const f32, n: usize, d: usize, k: usize, nt: i32, p: *mut i32, q: *mut i32, w: *mut f32) -> i32 {
    order_impl(data, n, d, k, nt, p, q, w)
}
#[no_mangle]
pub unsafe extern "C" fn dlmst_ivat_f64(data: *const f64, n: usize, d: usize, k: usize, nt: i32, out: *mut f64, p: *mut i32) -> i32 {
    ivat_impl(data, n, d, k, nt, out, p)
}
#[no_mangle]
pub unsafe extern "C" fn dlmst_ivat_f32(data: *const f32, n: usize, d: usize, k: usize, nt: i32, out: *mut f32, p: *mut i32) -> i32 {
    ivat_impl(data, n, d, k, nt, out, p)
}
#[no_mangle]
pub unsafe extern "C" fn prim_order_f64(data: *const f64, n: usize, nt: i32, p: *mut i32, q: *mut i32) -> i32 {
    prim_impl(data, n, nt, p, q)
}
#[no_mangle]
pub unsafe extern "C" fn prim_order_f32(data: *const f32, n: usize, nt: i32, p: *mut i32, q: *mut i32) -> i32 {
    prim_impl(data, n, nt, p, q)
}
