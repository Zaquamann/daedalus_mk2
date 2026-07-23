"""Cue-reliability heatmap: weight on audio vs (A reliability x V reliability).

Reads E1f_cue_weight_grid.csv. Heat = w_audio (0.5 = balanced, 1 = all audio,
0 = all video), diverging colormap centred at 0.5. Axes labelled by unisensory
accuracy (reliability). Cell text = w_audio; faint subscript = n_disagree.
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MSI = os.path.join(HERE, "analysis", "msi")
rows = list(csv.DictReader(open(os.path.join(MSI, "E1f_cue_weight_grid.csv"))))

sa = sorted({float(r["sigma_a"]) for r in rows})
sv = sorted({float(r["sigma_v"]) for r in rows})
W = np.full((len(sv), len(sa)), np.nan)
ND = np.zeros((len(sv), len(sa)), int)
aacc = {}
vacc = {}
for r in rows:
    i = sa.index(float(r["sigma_a"]))
    j = sv.index(float(r["sigma_v"]))
    W[j, i] = float(r["w_audio"])
    ND[j, i] = int(r["n_disagree"])
    aacc[i] = float(r["a_acc"]) * 100
    vacc[j] = float(r["v_acc"]) * 100

# order axes by ASCENDING reliability (low->high)
col = np.argsort([aacc[i] for i in range(len(sa))])
rowo = np.argsort([vacc[j] for j in range(len(sv))])
Wp = W[np.ix_(rowo, col)]
NDp = ND[np.ix_(rowo, col)]
xlab = [f"{aacc[i]:.0f}" for i in col]
ylab = [f"{vacc[j]:.0f}" for j in rowo]

fig, ax = plt.subplots(figsize=(7.2, 6.0))
im = ax.imshow(Wp, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(xlab)), xlab)
ax.set_yticks(range(len(ylab)), ylab)
ax.set_xlabel("Audio reliability  (A-only accuracy, %)")
ax.set_ylabel("Video reliability  (V-only accuracy, %)")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Weight on audio")
fig.tight_layout()
out = os.path.join(MSI, "FIG_cue_weight_heatmap.png")
fig.savefig(out, dpi=150)
fig.savefig(out.replace(".png", ".svg"))
plt.close(fig)

print("=== VALIDATION ===")
print("A reliability (cols, %):", xlab)
print("V reliability (rows, %):", ylab)
print("w_audio grid (rows=V low->high, cols=A low->high):")
for jj in range(Wp.shape[0] - 1, -1, -1):
    print("  " + "  ".join(f"{Wp[jj, ii]:.2f}" for ii in range(Wp.shape[1])))
print(f"wrote {out}")
