"""Held-out and finite-difference checks for the first self-calibration study."""
from pathlib import Path
import json
import numpy as np
from model import Budgets, effective_info, fisher, resources, score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'

BASE = np.array([1.4709697094658765, 1.4665247620626196, .1,
                 37.71228224598372, .5307443281912105, .95, 1.0])
SHAPED = np.array([1.479848418222164, 1.4664941080069658, .10004313477583075,
                   37.695841966048754, .530761948215107, .9499438090828508,
                   .9999467981979716, -.043096719884453405,
                   -9.072403480927368e-05])


def grid(bs, es, ds=(0.,)):
    return np.array([[d, b, e] for d in ds for b in bs for e in es])


def main():
    # The training grid is b=+/-0.3 and e=+/-0.1. These checks deliberately
    # leave that grid before making any claim about a robust advantage.
    grids = {
        'train': grid((-.3, 0., .3), (-.1, 0., .1)),
        'heldout': grid((-.6, -.45, -.15, .15, .45, .6),
                        (-.2, -.15, -.05, .05, .15, .2)),
        'delta_heldout': grid((-.45, 0., .45), (-.6, 0., .6),
                              ds=(-.01, 0., .01)),
    }
    rows = []
    for name, g in grids.items():
        for label, v, shaped in [('square', BASE, False),
                                 ('shaped', SHAPED, True)]:
            info = effective_info(fisher(g, v, shaped))
            rows.append(dict(grid=name, family=label, minimum=float(info.min()),
                             median=float(np.median(info)), maximum=float(info.max()),
                             n=int(len(g))))

    # Step convergence and nuisance rank at the training grid. The latter is
    # needed because Schur complements can look optimistic at singular points.
    steps = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    step_rows = []
    for label, v, shaped in [('square', BASE, False), ('shaped', SHAPED, True)]:
        for h in steps:
            info = effective_info(fisher(grids['train'], v, shaped, step=h))
            step_rows.append(dict(family=label, step=h, minimum=float(info.min())))
        f = fisher(grids['train'], v, shaped)
        ranks = [int(np.linalg.matrix_rank(z[1:, 1:], tol=1e-8)) for z in f]
        step_rows.append(dict(family=label, nuisance_ranks=ranks))

    result = dict(training_scores={
                      'square': float(score(BASE, False)),
                      'shaped': float(score(SHAPED, True))},
                  heldout=rows, finite_difference=step_rows,
                  resources={'square': resources(BASE), 'shaped': resources(SHAPED)})
    OUT.mkdir(exist_ok=True)
    (OUT/'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
