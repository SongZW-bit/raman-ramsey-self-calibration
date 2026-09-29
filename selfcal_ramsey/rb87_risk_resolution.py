"""Independent coupled correction to coarse Monte Carlo risk.

Estimate E[L_fine-L_coarse] with common OU paths and binomial uniforms.
Recompute calibration and branch selection at each temporal resolution.
"""
import argparse
import json
import time
import numpy as np
from scipy.stats import binom
from rb87_multilevel import RamanD1, probability_d1
from rb87_adaptive_experiment import Estimator
from rb87_multilevel_pilot import OUT
from two_beam_adaptive import posterior
from two_beam import atom_counts, geometry
from drift_study import layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", type=float, required=True)
    parser.add_argument("--reps", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--effective", action="store_true")
    parser.add_argument("--resolutions", type=int, nargs="+", choices=[1, 3], default=[1, 3])
    parser.add_argument("--seed", type=int, default=872194)
    args = parser.parse_args()
    lookup = "rb87_refined_effective_lookup.json" if args.effective else "rb87_refined_lookup.json"
    saved = json.loads((OUT / lookup).read_text())
    probability_fn = probability_d1
    if args.effective:
        from rb87_effective_compare import probability_effective
        probability_fn = probability_effective
    estimators = {e["name"]: Estimator(e, saved["grid"], saved["tables"][e["name"]]) for e in saved["designs"]}
    branches = [estimators[n] for n in ["branch_0", "branch_1", "branch_7"]]
    prefix = branches[0]
    grid = np.linspace(-.8, .8, 161)
    dg = np.linspace(-.003, .003, 21)
    means = prefix.mean(0., grid)[:, :2]
    cov = prefix.cov(grid)[:, :2, :2]
    jac = prefix.coeff(grid)[:, 1, :2]/.003
    risk = np.array([e.risk(grid) for e in branches])
    model = RamanD1()
    b, target = args.point, .0005
    seed = args.seed + round((b+.8)*10000)
    rng = np.random.default_rng(seed)
    times, shapes = [], []
    base = [estimators["fixed_shape"], prefix]
    for e in base:
        d, c = e.entry["design"], e.entry["config"]
        flags = layout(d["parameters"], d["pattern"], c)[0]
        centers = geometry(d["parameters"], d["family"], flags, c, 3)[0]
        coarse = geometry(d["parameters"], d["family"], flags, c, 1)[0]
        assert np.allclose(centers[:, 1::3], coarse, rtol=0, atol=1e-10)
        times.append(centers.ravel())
        shapes.append(centers.shape)
    c = prefix.entry["config"]
    assert all(e.entry["config"][key] == c[key] for e in base for key in
               ["sigma_intensity", "beam_correlation", "correlation_time"])
    union = np.unique(np.concatenate(times))
    path = np.empty((args.reps, 2, len(union)))
    std = c["sigma_intensity"]*np.sqrt(np.array([1+c["beam_correlation"], 1-c["beam_correlation"]])/2)
    path[:, :, 0] = rng.normal(size=(args.reps, 2))*std
    for j in range(1, len(union)):
        rr = np.exp(-(union[j]-union[j-1])/c["correlation_time"])
        path[:, :, j] = rr*path[:, :, j-1]+np.sqrt(1-rr*rr)*rng.normal(size=(args.reps, 2))*std
    noises = [path[:, :, np.searchsorted(union, t)].reshape(args.reps, 2, *shape).transpose(0, 2, 1, 3)
              for t, shape in zip(times, shapes)]
    uniforms = rng.uniform(1e-12, 1-1e-12, (args.reps, shapes[0][0]))
    result = dict(b=b, reps=args.reps, seed=seed, true_delta=target, rows=[], effective=args.effective,
                  scope="Independent paired fine-minus-coarse risk correction; same frozen D1 lookup; calibration and decisions recomputed at each resolution. Temporal substeps 1 and 3; not a continuum-limit certificate.")
    for steps in args.resolutions:
        start = time.perf_counter()
        for name, e, noise in zip(["fixed_shape", "adaptive"], base, noises):
            d, c = e.entry["design"], e.entry["config"]
            flags, phases = layout(d["parameters"], d["pattern"], c)[:2]
            counts = atom_counts(d["parameters"], d["family"], flags, c)
            nn = noise[..., 1::3] if steps == 1 else noise
            if name == "fixed_shape":
                p = probability_fn(target, b, nn, d["parameters"], d["family"], flags, phases, c, steps, model)
                fits = e.fit(binom.ppf(uniforms, counts, p)/counts)
                choices = None
            else:
                short = dict(c, frequency_offsets=np.asarray(c["frequency_offsets"])[:2].tolist(),
                             segment_phase_offsets=np.asarray(c["segment_phase_offsets"])[:2].tolist())
                pcal = probability_fn(target, b, nn[:, :2], d["parameters"], d["family"], flags[:2], phases[:2], short, steps, model)
                ycal = binom.ppf(uniforms[:, :2], counts[:2], pcal)/counts[:2]
                post = posterior(ycal, means, cov, jac, dg)
                choices = np.argmin(post @ risk.T, axis=1)
                fits = np.empty((args.reps, 2))
                for branch in np.unique(choices):
                    ids = np.flatnonzero(choices == branch)
                    selected = branches[branch]
                    p = probability_fn(target, b, nn[ids], d["parameters"], d["family"], flags, phases,
                                       selected.entry["config"], steps, model)
                    measured = binom.ppf(uniforms[ids, 2:], counts[2:], p[:, 2:])/counts[2:]
                    fits[ids] = selected.fit(np.column_stack((ycal[ids], measured)))
            row = dict(name=name, substeps=steps, mse=float(np.mean((fits[:, 0]-target)**2)),
                       estimates=fits.tolist(), choices=None if choices is None else choices.tolist(),
                       elapsed=time.perf_counter()-start)
            result["rows"].append(row)
            (OUT / args.output).write_text(json.dumps(result, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k not in ("estimates", "choices")}), flush=True)


if __name__ == "__main__":
    main()
