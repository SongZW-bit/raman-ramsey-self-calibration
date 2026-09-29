"""Reproduce GABRS properties, then optimize nested pulse families fairly."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize, root
from model import (BOUNDS, GRID, Budgets, effective_info, fisher, probabilities,
                   resources, rotate, score)

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'


def reproduce():
    v = [np.pi/2, np.pi/2, 1., 20., .5, .8, 1.]
    rows = []
    for b in np.linspace(-.8, .8, 17):
        # A common correction and a frequency solve both counted error signals.
        def error(z):
            p = probabilities([[z[0], b, 0.]], v, correction=z[1])[0]
            return p[:, 0]-p[:, 1]
        # With the Bloch sign convention used in model.py the compensating
        # phase is on the +2b branch near the nominal lock point.  Starting
        # on that branch avoids convergence to the neighboring Ramsey fringe.
        sol = root(error, [0., 2*b], options={'xtol': 1e-11})
        assert sol.success and np.linalg.norm(error(sol.x)) < 1e-9
        rows.append(dict(b=float(b), delta=float(sol.x[0]), correction=float(sol.x[1]),
                         residual=float(np.linalg.norm(error(sol.x)))))
    # Exact coherent rotation independently checked against Hilbert-space expm.
    rng = np.random.default_rng(2018)
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1., -1.]); pauli = np.array([sx, sy, sz])
    errors = []
    for _ in range(60):
        h = rng.normal(size=3); r = rng.normal(size=3); r /= np.linalg.norm(r)
        dt = rng.uniform(.01, 5.)
        rho = (np.eye(2)+np.einsum('i,ijk->jk', r, pauli))/2
        u = expm(-.5j*np.einsum('i,ijk->jk', h, pauli)*dt)
        evolved = u@rho@u.conj().T
        exact = np.einsum('ijk,kj->i', pauli, evolved).real
        errors.append(np.max(abs(exact-rotate(r, h, dt))))
    # Ideal two-time unknown phase calibration penalty, independently from FI.
    ts, tl, fl = 1., 20., .7
    cs, cl = np.exp(-.02*np.array([ts, tl]))
    f = sum(w*c*c*np.outer([t, 1.], [t, 1.])
            for t,c,w in [(ts,cs,1-fl),(tl,cl,fl)])
    direct = f[0,0]-f[0,1]**2/f[1,1]
    expected = (tl-ts)**2/(1/(fl*cl**2)+1/((1-fl)*cs**2))
    result = dict(gabrs=rows, max_rotation_error=float(max(errors)),
                  max_lock_shift=float(max(abs(row['delta']) for row in rows)),
                  ideal_effective_fisher=float(direct), ideal_formula=float(expected),
                  reproduction_scope='Analytical GABRS mechanism; not digitized article figures.')
    assert max(errors)<1e-12 and abs(direct-expected)<1e-10
    (OUT/'reproduction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Reproduction', result['max_lock_shift'], result['max_rotation_error'], flush=True)


def optimize(seeds, maxiter):
    rows=[]
    for shaped in (False, True):
        bounds=BOUNDS+([(-np.pi, np.pi)]*2 if shaped else [])
        for seed in seeds:
            start=time.monotonic()
            objective=lambda v: -score(v, shaped)/1e6
            result=differential_evolution(objective,bounds,seed=seed,maxiter=maxiter,
                                          popsize=8,tol=1e-6,polish=True,workers=1)
            # Retain a nested, locally refined conventional solution for shaped family.
            if shaped:
                baseline=max((r for r in rows if not r['shaped']),key=lambda r:r['score'])
                local=minimize(objective, baseline['parameters']+[0.,0.],
                               method='Powell', bounds=bounds, options={'maxiter':120})
                if local.fun<result.fun:result=local
            v=result.x
            row=dict(shaped=shaped,seed=seed,parameters=v.tolist(),score=score(v,shaped),
                     info_grid=effective_info(fisher(GRID,v,shaped)).tolist(),
                     resources=resources(v),seconds=time.monotonic()-start,
                     optimizer_success=bool(result.success),message=str(result.message))
            rows.append(row)
            (OUT/'optimization.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--seeds',type=int,nargs='+',default=[31,72,113])
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    reproduce();optimize(args.seeds,args.iterations)
