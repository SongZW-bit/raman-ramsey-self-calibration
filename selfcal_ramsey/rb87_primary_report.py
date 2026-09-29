"""Final primary comparison after covariance and temporal-resolution audits."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rb87_multilevel_pilot import OUT
from rb87_mc_report import summarize_adaptive


def main():
    parts = [json.loads((OUT / f"rb87_effective_fine_{side}1600.json").read_text())
             for side in ["negative", "positive"]]
    rows = []
    for part, b in zip(parts, [-.6, .6]):
        assert part["b"] == b and part["effective"] and part["reps"] == 1600
        assert len(part["rows"]) == 2 and all(r["substeps"] == 3 for r in part["rows"])
        assert {r["name"] for r in part["rows"]} == {"adaptive", "fixed_shape"}
        rows.extend([dict(r, b=b) for r in part["rows"]])
    merged = dict(effective=True, paired_randomness=True, reps=1600,
                  true_delta=parts[0]["true_delta"], rows=rows)
    qubit = summarize_adaptive(merged)
    qubit.update(scope="Independent confirmation at three temporal substeps, 1600 records per endpoint. Fixed frozen controls and covariance-safe likelihood; only fixed shaping and adaptive compared.")
    (OUT / "rb87_effective_fine_summary.json").write_text(json.dumps(qubit, indent=2))
    d1 = json.loads((OUT / "rb87_resolution_d1_summary.json").read_text())
    gain = [qubit["comparisons"][0]["mse_reduction"], d1["mse_reduction"]]
    bounds = np.array([qubit["comparisons"][0]["bootstrap_95_interval"], d1["bootstrap_95_interval"]])
    risks = [qubit["means"], d1["corrected_risks"]]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    colors = ["#6f7884", "#247e78"]
    for j, name in enumerate(["fixed_shape", "adaptive"]):
        axes[0].bar(np.arange(2)+(j-.5)*.32, [r[name]/1e-8 for r in risks], .30,
                    color=colors[j], label=["Fixed phase shaping", "Adaptive"][j])
    axes[0].set_xticks([0, 1], ["Effective qubit", "Full D1 manifold"])
    axes[0].set(ylabel="Frequency MSE / $10^{-8}$", title="Same frozen optical controls")
    axes[0].legend(frameon=False, fontsize=9)
    y = np.array(gain)*100
    axes[1].errorbar([0, 1], y, yerr=np.array([y-bounds[:, 0]*100, bounds[:, 1]*100-y]),
                    fmt="o", markersize=7, capsize=5, color="#b45f48")
    axes[1].axhline(0, color=".5", lw=.8)
    axes[1].set_xticks([0, 1], ["Effective qubit", "Full D1 manifold"])
    axes[1].set(xlim=(-.5, 1.5), ylabel="Adaptive MSE reduction (%)", title="95% Monte Carlo intervals")
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("1000 MHz; two nuisance endpoints; refined temporal sampling", fontsize=12)
    fig.savefig(OUT / "rb87_primary_comparison.png", dpi=220)
    fig.savefig(OUT / "rb87_primary_comparison.pdf")
    lines = ["# Cross-model adaptive Raman sensing: current evidence", "",
             "Frozen optical controls, 1000 MHz Hamiltonian detuning, equal-weight b=-0.6 and +0.6. The primary reference is the reoptimized fixed phase-shaped family. Both models use model-aware inference.", "",
             "| Model | Fixed MSE | Adaptive MSE | Reduction | 95% MC interval |",
             "|---|---:|---:|---:|---:|"]
    for i, label in enumerate(["Effective qubit", "Full 16-state D1"]):
        lines.append(f"| {label} | {risks[i]['fixed_shape']:.6g} | {risks[i]['adaptive']:.6g} | {100*gain[i]:.2f}% | [{100*bounds[i,0]:.2f}%, {100*bounds[i,1]:.2f}%] |")
    lines.extend(["", "## What was verified", "",
        "The qubit result is a fresh independent 1600-record-per-endpoint confirmation at three temporal samples per phase segment. The D1 result uses 200 coarse records plus an independent paired 64-record fine-minus-coarse correction per endpoint. The correction recomputes calibration and branch choice. Both bootstrap intervals include their actual sampling uncertainty.", "",
        "The D1 dynamics include both ground hyperfine manifolds, both D1 excited hyperfine manifolds and all Zeeman states, spontaneous emission, optical pumping and leakage. All 160000 prepared atoms count as resources; leakage is not postselected away. D2, atomic motion, polarization impurity and nonlinear Zeeman corrections are omitted.", "",
        "Covariance interpolation was audited. One effective-qubit fixed-shape covariance spline was nonpositive; it now uses log-Cholesky interpolation. The full-D1 splines were certified positive by recursive Bernstein subdivision and remained numerically unchanged. Earlier qubit results without certified/fine in their filenames are obsolete for this primary comparison.", "",
        "Eight-path conditional tests compare three and nine temporal samples; the largest conditional MSE difference was about 1.1%. This supports the refinement but is not a continuum-limit certificate. Confidence intervals above describe Monte Carlo uncertainty, not all atomic-model or numerical systematics.", "",
        "## Scope of the paper claim", "",
        "The result supports persistence of an adaptive design mechanism in a concrete multilevel model. It does not establish optimality across published pulses, all working detunings, or a continuous nuisance band. The controls were retrained in D1 and used in both models; likelihood tables were matched to each model. Blind use of a qubit likelihood on D1 data remains untested.", "",
        "A separate 200 MHz comparison shows that ordinary Ramsey benefits strongly from passive light-shift suppression. Its error is comparable with phase shaping there. The large passive detuning gain must not be attributed to adaptation. A useful next paper result is the boundary between passive suppression and calibration-conditioned pulse selection.", "",
        "The theoretical support is in ../Detuning_design_theory.md: interfering Raman paths, nuisance-aware information geometry, OU innovation covariance, the conditional value of calibration, and a sufficient estimator-risk stability margin. The current controller mainly learns a static beam ratio; it is not yet an explicit OU-state predictor.", ""])
    (OUT / "rb87_primary_report.md").write_text("\n".join(lines))
    print(json.dumps(dict(qubit=qubit, d1=d1), indent=2))


if __name__ == "__main__":
    main()
