"""Merge completed endpoint runs and show frozen-design model comparisons."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rb87_multilevel_pilot import OUT
from rb87_mc_report import summarize_adaptive


def main():
    pieces = [json.loads((OUT / name).read_text()) for name in
              ("rb87_refined_d1_negative200.json", "rb87_refined_d1_positive200.json")]
    for key in pieces[0]:
        if key != "rows":
            assert pieces[0][key] == pieces[1][key], key
    names = {"adaptive", "fixed_bank", "fixed_hyper", "fixed_shape"}
    for source, b in zip(pieces, [-.6, .6]):
        assert len(source["rows"]) == 4
        assert {r["name"] for r in source["rows"]} == names
        assert all(r["b"] == b and len(r["estimates"]) == source["reps"] for r in source["rows"])
    merged = dict(pieces[0], rows=pieces[0]["rows"]+pieces[1]["rows"])
    (OUT / "rb87_refined_d1_mc200.json").write_text(json.dumps(merged, indent=2))
    summary = summarize_adaptive(merged)
    summary.update(source="rb87_refined_d1_mc200.json", substeps=merged["substeps"],
                   scope="Frozen controls at 1000 MHz, equal-weight endpoints +/-0.6. Monte Carlo uncertainty only; temporal quadrature audited separately. No literature-optimality claim.")
    (OUT / "rb87_refined_d1_summary.json").write_text(json.dumps(summary, indent=2))
    qubit = json.loads((OUT / "rb87_refined_effective_certified_summary.json").read_text())
    labels = {"fixed_bank": "Fixed branch bank", "fixed_hyper": "Hyper-Ramsey family",
              "fixed_shape": "Fixed phase shaping", "adaptive": "Adaptive"}
    ordered = ["fixed_hyper", "fixed_shape", "fixed_bank", "adaptive"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), layout="constrained")
    colors = ["#247e78", "#b45f48"]
    for i, (report, title, color) in enumerate(zip([qubit, summary],
                   ["Effective qubit", "Full D1 manifold"], colors)):
        positions = np.arange(4)+(i-.5)*.34
        axes[0].bar(positions, [report["means"][n]/1e-8 for n in ordered],
                    width=.31, color=color, label=title)
        rows = [next(r for r in report["comparisons"] if r["reference"] == n) for n in ordered[:3]]
        y = np.array([r["mse_reduction"]*100 for r in rows])
        bounds = np.array([r["bootstrap_95_interval"] for r in rows])*100
        axes[1].errorbar(np.arange(3)+(i-.5)*.12, y,
                        yerr=np.array([y-bounds[:, 0], bounds[:, 1]-y]),
                        fmt="o", capsize=4, color=color, label=title)
    axes[0].set_xticks(np.arange(4), ["Hyper-Ramsey", "Phase-shaped", "Fixed bank", "Adaptive"], rotation=15)
    axes[0].set(ylabel="Frequency MSE / $10^{-8}$", title="Identical optical controls in both models")
    axes[1].set_xticks(np.arange(3), ["Hyper-Ramsey", "Phase-shaped", "Fixed bank"])
    axes[1].set(ylabel="Adaptive MSE reduction (%)", title="Paired bootstrap 95% intervals")
    axes[1].axhline(0, color=".5", lw=.8)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("1000 MHz; two endpoints; coarse pulse quadrature", fontsize=12)
    fig.savefig(OUT / "rb87_crossmodel_adaptive.png", dpi=220)
    fig.savefig(OUT / "rb87_crossmodel_adaptive.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
