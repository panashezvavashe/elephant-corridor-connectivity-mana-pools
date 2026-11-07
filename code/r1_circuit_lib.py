"""Raster circuit solver used to audit and rebuild the core-area network (#5, #10).

Reproduces Circuitscape 4's raster graph (eight neighbours, average resistance,
diagonal links scaled by sqrt 2, focal regions short-circuited) so that
  - Circuitscape's pairwise effective resistances can be re-derived and checked,
  - all ten cores can be shorted in ONE circuit and Kron-reduced to a 10-node
    network whose edge conductances and effective resistances are mutually
    consistent (the pairwise-mode matrix is not: each pair is a different circuit),
  - node-removal experiments can be run on that same circuit.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
import pyamg


def grid_conductances(res):
    """Edge list (i, j, g) over valid cells of a resistance grid, Circuitscape-style."""
    H, W = res.shape
    valid = np.isfinite(res) & (res > 0)
    idx = np.full(res.shape, -1, dtype=np.int64)
    idx[valid] = np.arange(valid.sum())
    I, J, G = [], [], []
    for dr, dc, scale in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2))):
        # pair cell (r, c) with (r + dr, c + dc)
        c0, c1 = max(0, -dc), W - max(0, dc)
        a, ra = idx[0:H - dr, c0:c1], res[0:H - dr, c0:c1]
        b, rb = idx[dr:H, c0 + dc:c1 + dc], res[dr:H, c0 + dc:c1 + dc]
        ok = (a >= 0) & (b >= 0)
        I.append(a[ok])
        J.append(b[ok])
        G.append(1.0 / (scale * (ra[ok] + rb[ok]) / 2.0))
    return idx, np.concatenate(I), np.concatenate(J), np.concatenate(G)


class Circuit:
    def __init__(self, res, regions):
        """res: 2-D resistance (NaN = nodata). regions: 2-D int, 0 = none, k = core k."""
        self.idx, self.I, self.J, self.G = grid_conductances(res)
        self.n_cells = int(self.idx.max() + 1)
        self.cell_region = np.zeros(self.n_cells, dtype=np.int64)
        m = self.idx >= 0
        self.cell_region[self.idx[m]] = regions[m]
        self.region_ids = sorted(int(k) for k in np.unique(regions[m]) if k > 0)

    def laplacian(self, shorted, extra_res=None):
        """Laplacian after contracting each region in `shorted` to a single node.
        Returns (L, node_of_region dict). extra_res: optional {region: resistance}
        overriding the resistance of that region's cells (habitat-loss scenario)."""
        I, J, G = self.I, self.J, self.G
        if extra_res:
            G = G.copy()
            for k, r in extra_res.items():
                hit = (self.cell_region[I] == k) | (self.cell_region[J] == k)
                G[hit] = np.minimum(G[hit], 1.0 / r)
        node = np.arange(self.n_cells)
        rep = {}
        for k in shorted:
            cells = np.where(self.cell_region == k)[0]
            node[cells] = cells[0]
            rep[k] = cells[0]
        uniq, node = np.unique(node, return_inverse=True)
        n = len(uniq)
        a, b = node[I], node[J]
        keep = a != b
        A = sp.coo_matrix((np.r_[G[keep], G[keep]], (np.r_[a[keep], b[keep]], np.r_[b[keep], a[keep]])),
                          shape=(n, n)).tocsr()
        L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A
        node_of = {k: int(node[rep[k]]) for k in rep}
        ncomp, lab = connected_components(A, directed=False)
        return L.tocsr(), node_of, lab

    @staticmethod
    def _solve(Lg, b):
        ml = pyamg.smoothed_aggregation_solver(Lg, symmetry='symmetric')
        return ml.solve(b, tol=1e-10, accel='cg', maxiter=500)

    def pairwise_resistance(self, k1, k2, short_all=False):
        """Circuitscape pairwise mode: only k1, k2 shorted (or all regions if short_all)."""
        L, node_of, lab = self.laplacian(self.region_ids if short_all else [k1, k2])
        s, g = node_of[k1], node_of[k2]
        comp = lab == lab[s]
        if lab[g] != lab[s]:
            return np.inf
        sub = np.where(comp)[0]
        pos = -np.ones(L.shape[0], dtype=np.int64)
        pos[sub] = np.arange(len(sub))
        Ls = L[sub][:, sub]
        keep = np.ones(len(sub), bool)
        keep[pos[g]] = False
        Lg = Ls[keep][:, keep].tocsr()
        b = np.zeros(Lg.shape[0])
        si = pos[s] - (1 if pos[s] > pos[g] else 0)
        b[si] = 1.0
        v = self._solve(Lg, b)
        return float(v[si])

    def kron_network(self, terminals, extra_res=None, shorted=None):
        """Short all `terminals` (plus any extra `shorted` regions) in one circuit and
        Kron-reduce onto the terminals. Returns (C, R): C = reduced conductance matrix
        (off-diagonal = direct-link conductance), R = effective-resistance matrix."""
        shorted = list(terminals) + [k for k in (shorted or []) if k not in terminals]
        L, node_of, lab = self.laplacian(shorted, extra_res)
        B = np.array([node_of[k] for k in terminals])
        main = lab == lab[B[0]]
        assert all(lab[b] == lab[B[0]] for b in B), 'terminals in different components'
        sub = np.where(main)[0]
        pos = -np.ones(L.shape[0], dtype=np.int64)
        pos[sub] = np.arange(len(sub))
        Ls = L[sub][:, sub].tocsr()
        Bp = pos[B]
        isB = np.zeros(len(sub), bool)
        isB[Bp] = True
        Ii = np.where(~isB)[0]
        LII = Ls[Ii][:, Ii].tocsr()
        LIB = Ls[Ii][:, Bp].toarray()
        LBB = Ls[Bp][:, Bp].toarray()
        ml = pyamg.smoothed_aggregation_solver(LII, symmetry='symmetric')
        Xs = np.column_stack([ml.solve(LIB[:, t], tol=1e-10, accel='cg', maxiter=500) for t in range(len(B))])
        Lred = LBB - LIB.T @ Xs
        Lred = (Lred + Lred.T) / 2
        Lp = np.linalg.pinv(Lred)
        d = np.diag(Lp)
        R = d[:, None] + d[None, :] - 2 * Lp
        return -Lred + np.diag(np.diag(Lred)), R
