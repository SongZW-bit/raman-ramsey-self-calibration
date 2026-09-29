"""Project nested OU quadrature differences through the frozen estimator.

Binomial replicates share uniforms across resolutions. They are conditional
replicates, not independent drift trajectories or a population-risk test.
"""
import json
import numpy as np
from scipy.stats import binom
from rb87_multilevel import RamanD1, probability_d1
from rb87_adaptive_experiment import Estimator
from rb87_multilevel_pilot import OUT
from two_beam_adaptive import trajectories
from two_beam import atom_counts


def main():
    saved = json.loads((OUT / "rb87_refined_lookup.json").read_text())
    model = RamanD1()
    rows = []
    target = .0005
    for name, b in [("fixed_hyper", -.6), ("fixed_shape", -.6),
                    ("branch_1", -.6), ("branch_7", .6)]:
        entry = next(e for e in saved["designs"] if e["name"] == name)
        d, c = entry["design"], entry["config"]
        est = Estimator(entry, saved["grid"], saved["tables"][name])
        rng = np.random.default_rng(691184)
        fine, flags, phases = trajectories(d, c, b, 8, rng, target, substeps=9)
        blocks = fine.reshape(8, len(flags), 2, 6, 9)
        counts = atom_counts(d["parameters"], d["family"], flags, c)
        uniforms = rng.uniform(1e-12, 1-1e-12, (8, 16, len(flags)))
        probabilities, fits = {}, {}
        for steps, ids in [(1, [4]), (3, [1, 4, 7]), (9, list(range(9)))]:
            noise = blocks[..., ids].reshape(8, len(flags), 2, 6*steps)
            p = probability_d1(target, b, noise, d["parameters"], d["family"],
                               flags, phases, c, steps, model)
            probabilities[steps] = p
            data = binom.ppf(uniforms, counts, p[:, None, :]) / counts
            fits[steps] = est.fit(data.reshape(-1, len(flags))).reshape(8, 16, 2)
        precision = np.linalg.inv(est.cov(b))
        a = est.coeff(b)[1] / .003
        g = (est.mean(0., b+1e-5)-est.mean(0., b-1e-5)) / 2e-5
        jac = np.column_stack((a, g))
        influence = np.linalg.solve(jac.T @ precision @ jac, jac.T @ precision)[0]
        row = dict(name=name, b=b, nested_trajectories=8,
                   conditional_binomial_replicates=16,
                   local_variance=float(est.risk(b)), comparisons=[])
        for steps in (1, 3):
            difference = probabilities[steps]-probabilities[9]
            projected = difference @ influence
            delta_difference = fits[steps][..., 0]-fits[9][..., 0]
            row["comparisons"].append(dict(
                substeps=steps, reference_substeps=9,
                max_whitened_squared_difference=float(np.einsum(
                    "ni,ij,nj->n", difference, precision, difference).max()),
                projected_difference_mean=float(projected.mean()),
                projected_difference_rms_over_local_sd=float(
                    np.sqrt(np.mean(projected**2)/est.risk(b))),
                fitted_difference_rms_over_local_sd=float(
                    np.sqrt(np.mean(delta_difference**2)/est.risk(b))),
                conditional_mse=float(np.mean((fits[steps][..., 0]-target)**2)),
                reference_conditional_mse=float(np.mean((fits[9][..., 0]-target)**2))))
        row["probabilities"] = {str(k): v.tolist() for k, v in probabilities.items()}
        row["fits"] = {str(k): v.tolist() for k, v in fits.items()}
        rows.append(row)
        result = dict(rows=rows, scope="Eight independent drift paths per design; 16 conditional detection replicates each. This tests sensitivity to quadrature, not population-MSE convergence or adaptive branch-selection convergence.")
        (OUT / "rb87_resolution_estimator.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in row.items() if k not in ("fits", "probabilities")}), flush=True)


if __name__ == "__main__":
    main()
