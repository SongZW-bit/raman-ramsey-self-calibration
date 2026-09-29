"""Independent two-level Monte Carlo estimator for time-refined risk."""
import argparse
import json
import numpy as np
from rb87_multilevel_pilot import OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--effective", action="store_true")
    args = parser.parse_args()
    if args.effective:
        base_name = "rb87_refined_effective_certified_mc400.json"
        correction_names = ["rb87_resolution_effective_certified_negative400.json", "rb87_resolution_effective_certified_positive400.json"]
        output = "rb87_resolution_effective_summary.json"
    else:
        base_name = "rb87_refined_d1_mc200.json"
        correction_names = ["rb87_resolution_negative64.json", "rb87_resolution_positive64.json"]
        output = "rb87_resolution_d1_summary.json"
    base = json.loads((OUT / base_name).read_text())
    corrections = [json.loads((OUT / n).read_text()) for n in correction_names]
    names = ["fixed_shape", "adaptive"]
    points = [-.6, .6]
    coarse, difference, fine_direct = {}, {}, {}
    for source, b in zip(corrections, points):
        assert source["b"] == b and source["true_delta"] == base["true_delta"]
        assert source["seed"] != base["seed"]
        assert source.get("effective", False) == args.effective and len(source["rows"]) == 4
    for name in names:
        coarse[name] = np.array([(np.array(next(r for r in base["rows"] if r["name"] == name and r["b"] == b)["estimates"])[:, 0]-base["true_delta"])**2 for b in points])
        pair = []
        for source in corrections:
            losses = {s: (np.array(next(r for r in source["rows"] if r["name"] == name and r["substeps"] == s)["estimates"])[:, 0]-base["true_delta"])**2 for s in (1, 3)}
            pair.append(losses)
        difference[name] = np.array([p[3]-p[1] for p in pair])
        fine_direct[name] = float(np.mean([p[3] for p in pair]))
    corrected = {n: float(coarse[n].mean()+difference[n].mean()) for n in names}
    rng = np.random.default_rng(517889)
    bootstrap = []
    for _ in range(10000):
        ix = rng.integers(0, coarse[names[0]].shape[1], coarse[names[0]].shape)
        jx = rng.integers(0, difference[names[0]].shape[1], difference[names[0]].shape)
        risks = {n: float(np.take_along_axis(coarse[n], ix, 1).mean()+np.take_along_axis(difference[n], jx, 1).mean()) for n in names}
        assert min(risks.values()) > 0
        bootstrap.append(1-risks["adaptive"]/risks["fixed_shape"])
    switches = []
    for source in corrections:
        choice = [np.array(next(r for r in source["rows"] if r["name"] == "adaptive" and r["substeps"] == s)["choices"]) for s in (1, 3)]
        switches.append(int(np.count_nonzero(choice[0] != choice[1])))
    result = dict(effective=args.effective, points=points, coarse_reps_per_point=base["reps"],
                  correction_reps_per_point=corrections[0]["reps"], coarse_source=base_name,
                  correction_sources=correction_names, corrected_substeps=3,
                  coarse_risks={n: float(coarse[n].mean()) for n in names},
                  mean_risk_correction={n: float(difference[n].mean()) for n in names},
                  corrected_risks=corrected, direct_fine_risks=fine_direct,
                  mse_reduction=1-corrected["adaptive"]/corrected["fixed_shape"],
                  bootstrap_95_interval=np.quantile(bootstrap, [.025, .975]).tolist(),
                  branch_switch_counts=switches,
                  scope="R3 estimated as independent mean(L1) + mean(L3-L1); paired uniforms and OU paths within correction, including new calibration decisions. Stratified bootstrap includes both independent samples. No continuum-limit or literature-optimality claim.")
    (OUT / output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
