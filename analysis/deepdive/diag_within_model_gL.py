#!/usr/bin/env python3
"""TEMP DEBUG (task #19) — recompute E1d gain-over-best with the SAME d'_AV_late
but two references: (a) cross-observer = standalone nets max(dA_std, dV_std) [the
committed benchmark], vs (b) within-model = the late net's OWN ablation channels
max(dA_fus_late, dV_fus_late). If the 'superoptimality' (gL>>1 in the tail) is the
D311 cross-observer artefact, swapping ONLY the reference collapses gL to a flat
<1 curve. Pure CSV arithmetic, no GPU, no model load.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "D312_ep150_e1d_reduced.csv")

print(f"source: {os.path.basename(CSV)}")
print(f"{'sigma':>6} {'dAVl':>6} {'dA_std':>7} {'dV_std':>7} "
      f"{'dAfL':>6} {'dVfL':>6} | {'gL_xobs':>8} {'gL_within':>9}")
rows = list(csv.DictReader(open(CSV)))
xobs, within = [], []
for r in rows:
    dAVl = float(r["dprime_AV_late"])
    dA_std = float(r["dprime_A_std"])
    dV_std = float(r["dprime_V_std"])
    dAfL = float(r["dA_fus_late"])
    dVfL = float(r["dV_fus_late"])
    g_x = dAVl / max(dA_std, dV_std)          # committed cross-observer reference
    g_w = dAVl / max(dAfL, dVfL)              # within-model (own channels)
    xobs.append(g_x)
    within.append(g_w)
    print(f"{float(r['sigma_a']):6.3f} {dAVl:6.3f} {dA_std:7.3f} {dV_std:7.3f} "
          f"{dAfL:6.3f} {dVfL:6.3f} | {g_x:8.3f} {g_w:9.3f}")

print(f"\ngL cross-observer : min={min(xobs):.3f} max={max(xobs):.3f} "
      f"(range {max(xobs) - min(xobs):.3f})  <- 'superoptimality' rises with sigma")
print(f"gL within-model   : min={min(within):.3f} max={max(within):.3f} "
      f"(range {max(within) - min(within):.3f})  <- FLAT; reference-dependence = artefact")
