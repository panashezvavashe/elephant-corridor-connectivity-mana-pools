"""Step 5: network audit and rebuild (#5, #10).

Usage: python r1_network.py <label> <cs_dir> <resistance.asc> <focal_nodes.asc> <resistances_3columns.out>

1. Core overlap / adjacency of the rasterized dBBMM 50% cores.
2. Circuitscape pairwise matrix: triangle-inequality violations (each pair is solved
   with only that pair short-circuited, so the matrix is not a metric).
3. Consistent network: all ten cores short-circuited in ONE circuit and Kron-reduced
   to ten nodes. Off-diagonal reduced conductances = direct links that do not pass
   through another core; effective resistances from the reduced Laplacian.
4. Centralities on that network (current-flow betweenness, shortest-path betweenness
   on link resistance, closeness on effective resistance).
5. Node removal for every core: (a) core no longer a node/short (landscape kept),
   (b) habitat loss: core cells set to the maximum resistance (1.0). Change in the
   effective resistance among the remaining nine cores and in network efficiency.
6. Sensitivity: cores whose rasterized footprints touch are merged into one node.
"""
import itertools
import os
import sys

import r1_config as C
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from r1_circuit_lib import Circuit

label, cs_dir, res_asc, node_asc, pair_out = sys.argv[1:6]
OUT = f'{C.OUT_TAB}/network_{label}'
os.makedirs(OUT, exist_ok=True)
ID_OF = {i + 1: cid for i, cid in enumerate(C.IDS)}  # node number -> collar (collar order)


def read_asc(p):
    with rasterio.open(p) as s:
        a = s.read(1).astype(float)
    return a


res = read_asc(res_asc)
res[res <= -9998] = np.nan
reg = read_asc(node_asc).astype(int)
reg[reg < 0] = 0
cell_km2 = 0.09 * 0.09 if res.shape[0] < 1500 else 0.03 * 0.03

# ── 1. Overlap / adjacency ────────────────────────────────────────────────────
cores = gpd.read_file(C.CORE_SHP).to_crs(C.CRS)
idcol = 'elephant_i' if 'elephant_i' in cores.columns else 'elephant_id'
cores = cores.set_index(idcol).loc[C.IDS]
rows = []
for a, b in itertools.combinations(range(1, 11), 2):
    ga, gb = cores.geometry.loc[ID_OF[a]], cores.geometry.loc[ID_OF[b]]
    touch_cells = int((ndimage.binary_dilation(reg == a, np.ones((3, 3))) & (reg == b)).sum())
    rows.append({'node_a': a, 'collar_a': ID_OF[a], 'node_b': b, 'collar_b': ID_OF[b],
                 'polygon_overlap_km2': ga.intersection(gb).area / 1e6,
                 'polygon_gap_km': ga.distance(gb) / 1000,
                 'raster_cells_in_contact': touch_cells})
adj = pd.DataFrame(rows)
size = pd.DataFrame({'node': range(1, 11), 'collar': C.IDS,
                     'polygon_km2': [cores.geometry.loc[c].area / 1e6 for c in C.IDS],
                     'raster_cells': [int((reg == k).sum()) for k in range(1, 11)]})
size['raster_km2'] = size.raster_cells * cell_km2
adj.round(4).to_csv(f'{OUT}/N1_core_overlap_adjacency.csv', index=False)
size.round(3).to_csv(f'{OUT}/N1b_core_sizes.csv', index=False)

# ── 2. Circuitscape pairwise matrix + triangle inequality ─────────────────────
R_cs = np.zeros((10, 10))
for line in open(pair_out):
    p = line.split()
    if len(p) == 3:
        i, j, r = int(float(p[0])), int(float(p[1])), float(p[2])
        R_cs[i - 1, j - 1] = R_cs[j - 1, i - 1] = r
pd.DataFrame(R_cs, index=C.IDS, columns=C.IDS).to_csv(f'{OUT}/N2_circuitscape_pairwise_resistance.csv')
viol = []
for i, j, k in itertools.permutations(range(10), 3):
    if i < j and R_cs[i, k] + R_cs[k, j] < R_cs[i, j]:
        viol.append({'i': C.IDS[i], 'j': C.IDS[j], 'via': C.IDS[k], 'R_ij': R_cs[i, j],
                     'R_ik+R_kj': R_cs[i, k] + R_cs[k, j]})
viol = pd.DataFrame(viol)
viol.to_csv(f'{OUT}/N3_triangle_violations.csv', index=False)
G_old = nx.Graph()
for i, j in itertools.combinations(range(10), 2):
    G_old.add_edge(C.IDS[i], C.IDS[j], weight=R_cs[i, j])
bt_old = nx.betweenness_centrality(G_old, weight='weight')

# ── 3-4. Consistent all-shorted network + centralities ────────────────────────
circ = Circuit(res, reg)
print('circuit built', circ.n_cells, 'cells', flush=True)
Cmat, R = circ.kron_network(list(range(1, 11)))
pd.DataFrame(Cmat, index=C.IDS, columns=C.IDS).to_csv(f'{OUT}/N4_kron_link_conductance.csv')
pd.DataFrame(R, index=C.IDS, columns=C.IDS).to_csv(f'{OUT}/N5_kron_effective_resistance.csv')


def centralities(Cm, R_, names):
    G = nx.Graph()
    n = len(names)
    for i, j in itertools.combinations(range(n), 2):
        if Cm[i, j] > 0:
            G.add_edge(names[i], names[j], conductance=Cm[i, j], link_res=1.0 / Cm[i, j])
    cfb = nx.current_flow_betweenness_centrality(G, weight='conductance', normalized=True)
    spb = nx.betweenness_centrality(G, weight='link_res', normalized=True)
    clo = {names[i]: (n - 1) / sum(R_[i, j] for j in range(n) if j != i) for i in range(n)}
    strength = {names[i]: sum(Cm[i, j] for j in range(n) if j != i) for i in range(n)}
    return pd.DataFrame({'current_flow_betweenness': cfb, 'shortest_path_betweenness': spb,
                         'closeness_effective_resistance': clo, 'node_strength_conductance': strength})


cent = centralities(Cmat, R, C.IDS)
cent['old_complete_graph_betweenness'] = pd.Series(bt_old)
for c in list(cent.columns[:4]):
    cent[f'rank_{c}'] = cent[c].rank(ascending=False, method='min').astype(int)
cent.index.name = 'collar'
cent['sex'] = cent.index.map(C.SEX)
cent.round(4).to_csv(f'{OUT}/N6_centrality_consistent_network.csv')
print(cent.round(3), flush=True)


# ── 5. Node removal ───────────────────────────────────────────────────────────
def eff(Rm):
    iu = np.triu_indices(len(Rm), 1)
    return float(np.mean(1.0 / Rm[iu]))


def node_removal(circuit, R_full, labels, node_ids):
    """Remove each node in turn; compare effective resistance among the remaining nodes.
    Primary metric is the mean % increase in R; efficiency (mean 1/R) is reported too but is
    dominated by near-zero resistances between touching cores."""
    out = []
    n = len(node_ids)
    for pos_k, k in enumerate(node_ids):
        others = [o for o in node_ids if o != k]
        oi = [p for p in range(n) if p != pos_k]
        base = R_full[np.ix_(oi, oi)]
        iu = np.triu_indices(n - 1, 1)
        for scen, kw in (('core not shorted (landscape kept)', {}),
                         ('habitat loss: core cells at max resistance', {'extra_res': {k: 1.0}})):
            _, Rk = circuit.kron_network(others, **kw)
            ratio = Rk[iu] / base[iu]
            worst = int(np.argmax(ratio))
            lab_o = [labels[p] for p in oi]
            out.append({'removed': labels[pos_k], 'scenario': scen,
                        'mean_R_increase_pct': 100 * (ratio.mean() - 1),
                        'median_R_increase_pct': 100 * (np.median(ratio) - 1),
                        'max_R_increase_pct': 100 * (ratio.max() - 1),
                        'most_affected_pair': f'{lab_o[iu[0][worst]]} / {lab_o[iu[1][worst]]}',
                        'efficiency_change_pct': 100 * (eff(Rk) / eff(base) - 1)})
        print('removed', labels[pos_k], flush=True)
    df = pd.DataFrame(out)
    df['rank_by_mean_R_increase'] = df.groupby('scenario')['mean_R_increase_pct'].rank(
        ascending=False, method='min').astype(int)
    return df


if not os.path.exists(f'{OUT}/N7_node_removal.csv'):
    node_removal(circ, R, [str(c) for c in C.IDS], list(range(1, 11))).round(4).to_csv(
        f'{OUT}/N7_node_removal.csv', index=False)

# ── 6. Merge touching cores into single nodes ─────────────────────────────────
Gt = nx.Graph()
Gt.add_nodes_from(range(1, 11))
for _, r_ in adj[adj.raster_cells_in_contact > 0].iterrows():
    Gt.add_edge(int(r_.node_a), int(r_.node_b))
clusters = [sorted(c) for c in nx.connected_components(Gt)]
reg_m = np.zeros_like(reg)
names = []
for ci, cl in enumerate(sorted(clusters), start=1):
    reg_m[np.isin(reg, cl)] = ci
    names.append('+'.join(str(ID_OF[k]) for k in cl))
circ_m = Circuit(res, reg_m)
Cm, Rm = circ_m.kron_network(list(range(1, len(clusters) + 1)))
cent_m = centralities(Cm, Rm, names)
cent_m.index.name = 'merged_node'
cent_m.round(4).to_csv(f'{OUT}/N8_centrality_merged_touching_cores.csv')
print(cent_m.round(3))
node_removal(circ_m, Rm, names, list(range(1, len(clusters) + 1))).round(4).to_csv(
    f'{OUT}/N9_node_removal_merged_touching_cores.csv', index=False)
print('violations', len(viol), 'of', 10 * 9 * 8 // 2)
print('done')
