"""Retune square-pulse designs across correlation times and audit dense grids."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import differential_evolution, minimize
from drift_study import evaluate, layout
from drift_continuous import statistics

OUT = Path(__file__).resolve().parent / 'results'
BOUNDS = [(.6,5.),(.6,7.),(.1,5.),(10.,120.),(.6,1.),
          (-np.pi,np.pi),(-1.,1.),(-np.pi,np.pi)]


def polish(v, pattern, config):
    active = list(np.linspace(-.6,.6,13))
    for cycle in range(3):
        def constraints(y):
            # The evaluator's budget penalty must not replace physical values
            # during constrained optimization just outside the feasible region.
            unlimited = dict(config,time_cap=1e8,exposure_cap=1e8)
            records = evaluate(y[:-1],'square',pattern,unlimited,active,True)['records']
            cost = layout(y[:-1],pattern,config)[-1]
            return np.r_[y[-1]-np.array([r['variance']/1e-8 for r in records]),
                         (config['time_cap']-cost['elapsed'])/100,
                         (config['exposure_cap']-cost['exposure'])/10]
        initial = np.r_[v,evaluate(v,'square',pattern,config,active)/1e-8]
        fit = minimize(lambda y:y[-1],initial,method='SLSQP',
                       bounds=BOUNDS+[(.001,10000.)],
                       constraints=[dict(type='ineq',fun=constraints)],
                       options=dict(maxiter=100,ftol=1e-8,eps=2e-6))
        if min(constraints(fit.x)) >= -1e-6 and fit.fun < initial[-1]:
            v = fit.x[:-1].copy()
            # Put numerical boundary residuals on the feasible side.
            cost = layout(v,pattern,config)[-1]
            if cost['elapsed'] > config['time_cap']:
                nlong = sum(layout(v,pattern,config)[0])
                v[3] -= (cost['elapsed']-config['time_cap']+1e-9)/nlong
        grid = np.linspace(-.6,.6,81)
        records = evaluate(v,'square',pattern,config,grid,True)['records']
        active = sorted(set(active + [r['b'] for r in sorted(records,key=lambda r:r['variance'])[-3:]]))
    return list(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--taus',type=float,nargs='+',default=[300.,1000.,3000.])
    parser.add_argument('--iterations',type=int,default=120)
    parser.add_argument('--atoms',type=int,default=10000)
    parser.add_argument('--tag',default='')
    args = parser.parse_args()
    source = json.loads((OUT/'drift_schedules.json').read_text())
    config = dict(source['config'],atoms_per_shot=args.atoms)
    baselines = {m:json.loads((OUT/f'drift_static_{m}.json').read_text())['results']
                 for m in ('quasistatic','shot')}
    output = OUT/f'drift_correlation_scan{args.tag}.json'
    rows=[]
    training = np.linspace(-.6,.6,13)
    for mode in ('quasistatic','shot','OU'):
        times = args.taus if mode=='OU' else [1e9 if mode=='quasistatic' else 100.]
        for tau in times:
            design_config = dict(config,correlation_time=tau)
            if mode=='shot':design_config['noise_model']='shot'
            for count in (8,10,12):
                pattern=f'distributed{count}'
                candidates=[r['parameters'] for r in source['results']
                            if r['family']=='square' and r['pattern']==pattern]
                candidates += [r['parameters'] for rs in baselines.values() for r in rs if r['pattern']==pattern]
                candidates += [r['parameters'] for r in rows if r['pattern']==pattern]
                start=min(candidates,key=lambda v:evaluate(v,'square',pattern,design_config,training))
                for seed in (223,661):
                    fit=differential_evolution(lambda v:evaluate(v,'square',pattern,design_config,training)/1e-8,
                        BOUNDS,x0=start,seed=seed,maxiter=args.iterations,popsize=7,tol=1e-6,polish=False)
                    candidates.append(fit.x.tolist())
                best=min(candidates,key=lambda v:evaluate(v,'square',pattern,design_config,training))
                v=polish(best,pattern,design_config)
                actual=[]
                for test_tau in args.taus:
                    cfg=dict(config,correlation_time=test_tau)
                    checks=[statistics(v,'square',pattern,b,cfg,3) for b in np.linspace(-.6,.6,81)]
                    worst=max(checks,key=lambda z:z['variance'])
                    actual.append(dict(tau=test_tau,variance=worst['variance'],shot=worst['shot_variance'],drift=worst['drift_variance']))
                row=dict(mode=mode,design_tau=tau,family='square',pattern=pattern,parameters=v,
                         actual=actual,resources=layout(v,pattern,config)[-1])
                rows.append(row)
                output.write_text(json.dumps(dict(config=config,taus=args.taus,results=rows),indent=2),encoding='utf-8')
                print(json.dumps({k:z for k,z in row.items() if k!='parameters'}),flush=True)


if __name__=='__main__':main()
