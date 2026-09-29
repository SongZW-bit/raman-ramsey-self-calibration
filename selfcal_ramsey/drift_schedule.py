"""Refine shot allocation and longer dark times for every pulse family."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import differential_evolution, minimize
from drift_study import evaluate, layout
from drift_continuous import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    src=json.loads((OUT/'drift_refined_quadrature.json').read_text())
    config=src['config'];bs=np.linspace(-.6,.6,13)
    previous=OUT/'drift_schedules.json'
    rows=json.loads(previous.read_text())['results'] if previous.exists() else []
    bounds=[(.6,5.),(.6,7.),(.1,5.),(10.,120.),(.6,1.),(-np.pi,np.pi)]
    for kind in ('square','hyper','shape'):
        src=json.loads((OUT/'drift_refined_quadrature.json').read_text())
        for nshort in (8,10,12):
            pattern='distributed'+str(nshort)
            if any(z['family']==kind and z['pattern']==pattern for z in rows):continue
            prior=min((z for z in src['results'] if z['family']==kind),key=lambda z:z['variance'])
            v=prior['parameters']
            bounds_kind=bounds+([(-np.pi,np.pi)]*4 if kind=='shape' else [])+[(-1.,1.),(-np.pi,np.pi)]
            candidates=[v]
            if kind=='shape':
                z=min((r for r in rows if r['pattern']==pattern),key=lambda r:r['variance'])
                candidates.append(z['parameters'][:6]+([0.,0.,np.pi,np.pi] if z['family']=='hyper' else [0.]*4)+z['parameters'][6:])
            for seed in (941,1583):
                fit=differential_evolution(lambda v:evaluate(v,kind,pattern,config,bs)/1e-8,
                                           bounds_kind,x0=candidates[-1],seed=seed,
                                           maxiter=220,popsize=7,tol=2e-6)
                candidates.append(fit.x.tolist())
            v=min(candidates,key=lambda v:evaluate(v,kind,pattern,config,bs))
            # Compact 3-point local polish; dense verification remains separate.
            active=list(bs)
            def vals(x):return np.array([statistics(x,kind,pattern,b,config,2)['variance']/1e-8 for b in active])
            for cycle in range(2):
                def cons(y):
                    cost=layout(y[:-1],pattern,config)[-1]
                    return np.r_[y[-1]-vals(y[:-1]),(config['time_cap']-cost['elapsed'])/100,
                                 (config['exposure_cap']-cost['exposure'])/10]
                init=np.r_[v,max(vals(v))]
                fit=minimize(lambda y:y[-1],init,method='SLSQP',bounds=bounds_kind+[(.01,1000.)],
                             constraints=[dict(type='ineq',fun=cons)],options=dict(maxiter=45,ftol=1e-7,eps=2e-5))
                if min(cons(fit.x))>=-1e-5 and fit.fun<init[-1]:v=fit.x[:-1].tolist()
                dense=np.linspace(-.6,.6,61)
                checks=[statistics(v,kind,pattern,b,config,3) for b in dense]
                indices=np.argsort([c['variance'] for c in checks])[-3:]
                active=sorted(set(active+[float(dense[k]) for k in indices]))
            worst=max(checks,key=lambda z:z['variance'])
            row=dict(family=kind,pattern=pattern,parameters=v,variance=worst['variance'],
                     resources=worst['resources'],dense_variances=[z['variance'] for z in checks])
            rows.append(row)
            (OUT/'drift_schedules.json').write_text(json.dumps(dict(config=config,results=rows),indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k!='dense_variances'}),flush=True)


if __name__=='__main__':main()
