"""Check interpolation, nesting, and a freely optimized readout-phase baseline."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar
from model import BOUNDS, GRID, effective_info, fisher, probabilities, resources, score

OUT = Path(__file__).resolve().parent / 'results'


def main():
    prior = json.loads((OUT/'optimization.json').read_text())
    chosen = [max((r for r in prior if r['shaped'] == shaped), key=lambda r:r['score'])
              for shaped in (False, True)]
    # Optimization needs an in-domain validation grid. Extrapolation alone
    # cannot diagnose overfitting within the specified uncertainty set.
    dense = np.array([[0., b, e] for b in np.linspace(-.3, .3, 31)
                     for e in np.linspace(-.1, .1, 21)])
    rows = []
    for r in chosen:
        v, shaped = r['parameters'], r['shaped']
        f = fisher(dense, v, shaped)
        info = effective_info(f)
        k = int(np.argmin(info))
        rows.append(dict(family='shaped' if shaped else 'square',
                         minimum=float(info[k]), worst_point=dense[k].tolist(),
                         parameters=v, resources=resources(v)))
    v0 = chosen[0]['parameters']
    nested_error = np.max(abs(probabilities(dense, v0)-
                              probabilities(dense, v0+[0.,0.], True)))
    # Readout phase is an inexpensive control and must be available to the
    # square baseline if an internal pulse phase is to count as an improvement.
    offset = minimize_scalar(lambda phi:-score(v0+[phi]), bounds=(-.2,.2),
                             method='bounded', options={'xatol':1e-9})
    start = v0+[float(offset.x)]
    fit = differential_evolution(lambda v:-score(v)/1e6,
                                 BOUNDS+[(-.25,.25)], x0=start,
                                 seed=2026, maxiter=160, popsize=8, tol=1e-7)
    if score(fit.x)<score(start):
        best=start
    else:
        best=fit.x.tolist()
    f=fisher(dense,best)
    info=effective_info(f)
    k=int(np.argmin(info))
    rows.append(dict(family='square_with_readout_offset',
                     minimum=float(info[k]), worst_point=dense[k].tolist(),
                     training_minimum=score(best), parameters=best,
                     resources=resources(best), optimizer_success=bool(fit.success)))
    result=dict(dense_grid_size=len(dense), nested_probability_error=float(nested_error),
                designs=rows,
                note='All families have fixed controls; no true nuisance values are supplied to the design.')
    (OUT/'design_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
