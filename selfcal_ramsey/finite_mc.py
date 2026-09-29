"""Small finite-data check of the joint (delta, b, e) estimator."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import minimize
from model import PHASES, probabilities, resources, weights

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'

BASE = np.array([1.4709697094658765, 1.4665247620626196, .1,
                 37.71228224598372, .5307443281912105, .95, 1.0])
SHAPED = np.array([1.479848418222164, 1.4664941080069658, .10004313477583075,
                   37.695841966048754, .530761948215107, .9499438090828508,
                   .9999467981979716, -.043096719884453405,
                   -9.072403480927368e-05])


def run_family(v, shaped, true_x, reps=120, seed=20260918):
    rng = np.random.default_rng(seed)
    # Round every allocation down so none of the resource caps is exceeded.
    ns = np.floor(resources(v)['atoms'] * weights(v)).astype(int)
    n = int(ns.sum())
    p0 = probabilities([true_x], v, shaped)[0]
    bounds = [(-.03, .03), (-1.2, 1.2), (-.35, .35)]
    estimates = []
    failures = 0
    for _ in range(reps):
        counts = rng.binomial(ns, p0)

        def nll(x):
            p = probabilities([x], v, shaped)[0]
            return float(-np.sum(counts*np.log(p) + (ns-counts)*np.log1p(-p)))

        fit = minimize(nll, np.zeros(3), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        failures += int(not fit.success)
        estimates.append(fit.x)
    arr = np.asarray(estimates)
    return dict(family='shaped' if shaped else 'square', true_x=true_x.tolist(),
                n=n, reps=reps, failures=failures,
                mean=arr.mean(axis=0).tolist(), bias=(arr.mean(axis=0)-true_x).tolist(),
                covariance=np.cov(arr, rowvar=False, ddof=1).tolist(),
                delta_std=float(arr[:, 0].std(ddof=1)),
                delta_rmse=float(np.sqrt(np.mean((arr[:, 0]-true_x[0])**2))),
                scope='Estimator smoke check; 120 trials cannot resolve a one-percent precision gain.')


def main():
    # The local design point where the shaped family had its largest training
    # grid advantage, plus the edge of that training grid.
    rows = []
    for true_x, seed in [([0., .3, .1], 20260918), ([0., -.3, -.1], 20260919)]:
        rows.append(run_family(BASE, False, np.asarray(true_x, float), seed=seed))
        rows.append(run_family(SHAPED, True, np.asarray(true_x, float), seed=seed+100))
    OUT.mkdir(exist_ok=True)
    (OUT/'finite_mc.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
