"""Generate + self-validate two MSI figures from the verified source CSVs.

Tufte-minimal: no title, no gridlines, left/bottom spines only, no annotations.
Every plotted value is read straight from the CSV (nothing hard-coded) and echoed
for cell-by-cell validation against the file.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MSI = os.path.join(HERE, "analysis", "msi")
BLUE, GREEN, RED, ORANGE = "#4C72B0", "#55A868", "#C44E52", "#DD8452"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def tufte(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=3)


# ---------------------------------------------------------------- Fig 1: bars
e1e = read_csv(os.path.join(MSI, "E1e_matched_75.csv"))
row = next(r for r in e1e if r["condition"] == "matched_75")
A, V, AV, Pavg, Pbay = (float(row[k]) * 100 for k in
                        ("A_acc", "V_acc", "AV_acc", "pooled_avg", "pooled_bayes"))

fig, ax = plt.subplots(figsize=(7.0, 4.6))
labels = ["Audio", "Video", "Audio-visual", "Pooled\n(avg)", "Pooled\n(Bayes)"]
vals = [A, V, AV, Pavg, Pbay]
bars = ax.bar(labels, vals, color=[BLUE, GREEN, RED, "#BDBDBD", "#757575"], width=0.66)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
            ha="center", va="bottom", fontsize=10.5)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
tufte(ax)
fig.tight_layout()
out1 = os.path.join(MSI, "FIG_msi_bars_noisy.png")
fig.savefig(out1, dpi=150)
fig.savefig(out1.replace(".png", ".svg"))
plt.close(fig)


# ---------------------------------------------------------------- Fig 2: line
e1 = read_csv(os.path.join(MSI, "E1_inverse_effectiveness.csv"))
sig = [float(r["sigma_per_rms"]) for r in e1]
a_acc = [float(r["A_acc"]) * 100 for r in e1]
av_acc = [float(r["AV_acc"]) * 100 for r in e1]
gain = [float(r["AV_minus_A"]) * 100 for r in e1]
x = list(range(len(sig)))

fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(x, a_acc, "o-", color=BLUE, lw=1.8, ms=5, label="Audio-only")
ax.plot(x, av_acc, "s-", color=RED, lw=1.8, ms=5, label="Audio-visual")
ax.plot(x, gain, "^--", color=ORANGE, lw=1.8, ms=5, label="AV − A gain")
ax.set_xticks(x)
ax.set_xticklabels([f"{s:g}" for s in sig])
ax.set_xlabel("Audio noise  σ  (per-RMS)")
ax.set_ylabel("Accuracy / gain (%)")
ax.set_ylim(0, 100)
tufte(ax)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
out2 = os.path.join(MSI, "FIG_inverse_effectiveness_line.png")
fig.savefig(out2, dpi=150)
fig.savefig(out2.replace(".png", ".svg"))
plt.close(fig)


# ---------------------------------------------------------------- validation
print("=== VALIDATION — plotted values vs source CSV cells ===")
print(f"FIG 1 {out1}  (E1e matched_75: sigma_a={row['sigma_a']} sigma_v={row['sigma_v']})")
print(f"  A {A:.3f}  V {V:.3f}  AV {AV:.3f}  Pavg {Pavg:.3f}  Pbay {Pbay:.3f}")
print(f"  (csv A/V/AV/Pavg/Pbay = {float(row['A_acc']):.6f}/{float(row['V_acc']):.6f}/"
      f"{float(row['AV_acc']):.6f}/{float(row['pooled_avg']):.6f}/{float(row['pooled_bayes']):.6f})")
print(f"  genuine MSI: AV-Pavg = +{AV-Pavg:.2f}pp ; AV-Pbay = +{AV-Pbay:.2f}pp")
print(f"FIG 2 {out2}  (E1_inverse_effectiveness.csv)")
for r in e1:
    print(f"  sig {float(r['sigma_per_rms']):6.4f}  A {float(r['A_acc'])*100:6.3f}  "
          f"AV {float(r['AV_acc'])*100:6.3f}  gain {float(r['AV_minus_A'])*100:6.3f}")
