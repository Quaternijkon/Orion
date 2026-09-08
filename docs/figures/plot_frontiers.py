"""Render the matched-recall QPS-vs-Recall Pareto frontiers for the five serving
arms across four datasets (2x2 grid). Data are the measured points transcribed
from experiments/RESULTS.md (GloVe/SIFT/Deep) and the DBpedia runs. Output:
frontiers.pdf (for the paper) and frontiers.png (for preview)."""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (recall, qps) points per arm; frontiers are drawn through the upper envelope.
DATA = {
    "SIFT-128-euclidean\n(image descriptors, near-uniform)": {
        "Orion navigation": [(0.9781, 5523), (0.9901, 3865), (0.9952, 2955)],
        "k-means centroid": [(0.9858, 4452), (0.9968, 2152), (0.9973, 1290), (0.9974, 883)],
        "hash broadcast":   [(0.9990, 1025), (0.9993, 685), (0.9994, 444)],
        "k-means broadcast":[(0.9914, 1030), (0.9974, 643), (0.9991, 386)],
        "Orion broadcast":  [(0.9915, 1167), (0.9976, 786), (0.9989, 493)],
        "target": 0.99,
    },
    "DBpedia-1536-angular\n(OpenAI ada-002 text)": {
        "Orion navigation": [(0.9645, 1004), (0.9814, 602), (0.9883, 415)],
        "k-means centroid": [(0.9797, 498), (0.9863, 328), (0.9931, 204), (0.9957, 135), (0.9979, 81)],
        "hash broadcast":   [(0.9966, 154), (0.9987, 95), (0.9995, 58)],
        "k-means broadcast":[(0.9894, 147), (0.9962, 86), (0.9984, 52)],
        "Orion broadcast":  [(0.9888, 139), (0.9955, 84), (0.9980, 50)],
        "target": 0.98,
    },
    "GloVe-200-angular\n(word vectors, angular)": {
        "Orion navigation": [(0.8390, 1994), (0.8804, 1286), (0.9069, 974)],
        "k-means centroid": [(0.8896, 809), (0.9220, 500), (0.9379, 331)],
        "hash broadcast":   [(0.9435, 418), (0.9777, 251), (0.9941, 151)],
        "k-means broadcast":[(0.8924, 397), (0.9456, 244), (0.9765, 146)],
        "Orion broadcast":  [(0.8875, 439), (0.9373, 276), (0.9684, 170)],
        "target": 0.90,
    },
    "Deep-96-angular\n(learned CNN embeddings)": {
        "Orion navigation": [(0.9740, 6155), (0.9873, 4328), (0.9923, 3209)],
        "k-means centroid": [(0.9577, 3530), (0.9842, 1537), (0.9864, 975), (0.9868, 686)],
        "hash broadcast":   [(0.9959, 956), (0.9986, 628), (0.9994, 395)],
        "k-means broadcast":[(0.9658, 796), (0.9868, 496), (0.9952, 288)],
        "Orion broadcast":  [(0.9642, 1070), (0.9863, 650), (0.9951, 394)],
        "target": 0.98,
    },
}

STYLE = {
    "Orion navigation":  dict(color="#d62728", marker="o", lw=2.2, ms=6, zorder=5),
    "k-means centroid":  dict(color="#1f77b4", marker="s", lw=1.6, ms=5),
    "hash broadcast":    dict(color="#7f7f7f", marker="^", lw=1.2, ms=4, ls="--"),
    "k-means broadcast": dict(color="#2ca02c", marker="v", lw=1.2, ms=4, ls="--"),
    "Orion broadcast":   dict(color="#ff7f0e", marker="D", lw=1.2, ms=4, ls="--"),
}

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
axes = axes.ravel()
for ax, (title, arms) in zip(axes, DATA.items()):
    target = arms.get("target")
    for arm, style in STYLE.items():
        pts = sorted(arms[arm])  # by recall
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, label=arm, **style)
    if target is not None:
        ax.axvline(target, color="k", ls=":", lw=1.0, alpha=0.5)
        ax.text(target, ax.get_ylim()[1], f" R={target:g}", va="top", ha="left",
                fontsize=7, color="k", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Recall@10", fontsize=8)
    ax.set_ylabel("QPS (log)", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.tick_params(labelsize=7)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 1.005))
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("frontiers.pdf", bbox_inches="tight")
fig.savefig("frontiers.png", dpi=150, bbox_inches="tight")
print("wrote frontiers.pdf and frontiers.png")
