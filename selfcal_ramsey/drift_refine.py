"""Give all families frequency compensation and independent short readout phase."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import differential_evolution, minimize
from drift_study import evaluate
from drift_continuous import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--width',type=float)
    parser.add_argument('--tag',default='')
    parser.add_argument('--iterations',type=int,default=250)
    parser.add_argument('--quadrature',action='store_true')
    parser.add_argument('--atoms',type=int,default=10000)
    parser.add_argument('--patterns',nargs='+',default=['alternating','abba','pairs'])
    parser.add_argument('--maxpulse',type=float,default=7.)
    parser.add_argument('--exposure',type=float,default=160.)
    args=parser.parse_args()
    source=json.loads((OUT/'drift_deadtime_91.json').read_text())
    config=source['config']
    config['quadrature']=args.quadrature
    config['atoms_per_shot']=args.atoms
    config['exposure_cap']=args.exposure
    bs=(.2,.3,.4) if args.width is None else tuple(np.linspace(-args.width,args.width,13))
    bounds=[(.6,args.maxpulse),(.6,args.maxpulse),(.1,5.),(10.,120.),(.6,1.),(-np.pi,np.pi)]
    rows=[]
    for kind in ('square','hyper','shape'):
        for pattern in args.patterns:
            prior=min((r for r in source['results'] if r['family']==kind),key=lambda r:r['variance'])
            v=prior['parameters']+[0.,0.]
            b=bounds+([(-np.pi,np.pi)]*4 if kind=='shape' else [])+[(-1.,1.),(-np.pi,np.pi)]
            candidates=[v]
            if kind=='shape':
                base=min((r for r in rows if r['pattern']==pattern),key=lambda r:r['variance'])
                candidates.append(base['parameters'][:6]+([0.,0.,np.pi,np.pi] if base['family']=='hyper' else [0.]*4)+base['parameters'][6:])
            for seed in (137,281):
                fit=differential_evolution(lambda v:evaluate(v,kind,pattern,config,bs)/1e-8,b,
                                           x0=candidates[-1],seed=seed,maxiter=args.iterations,popsize=7,tol=2e-6)
                candidates.append(fit.x.tolist())
            v=min(candidates,key=lambda v:evaluate(v,kind,pattern,config,bs))
            # Refine using noise resolved within each pulse. Epigraph avoids
            # taking finite differences across a nonsmooth worst-case maximum.
            def vals(v):
                return np.array([statistics(v,kind,pattern,x,config,2)['variance']/1e-8 for x in bs])
            def constraints(y):
                cost=evaluate(y[:-1],kind,pattern,config,details=True)['resources']
                return np.r_[y[-1]-vals(y[:-1]),(config['time_cap']-cost['elapsed'])/100,
                             (config['exposure_cap']-cost['exposure'])/10]
            init=np.r_[v,vals(v).max()]
            fit=minimize(lambda y:y[-1],init,method='SLSQP',bounds=b+[(.01,1000.)],
                         constraints=[dict(type='ineq',fun=constraints)],
                         options=dict(maxiter=100,ftol=1e-8,eps=2e-5))
            if min(constraints(fit.x))>=-1e-6 and fit.fun<init[-1]:v=fit.x[:-1].tolist()
            checks=[statistics(v,kind,pattern,x,config,8) for x in np.linspace(min(bs),max(bs),41)]
            worst=max(checks,key=lambda z:z['variance'])
            row=dict(family=kind,pattern=pattern,parameters=v,variance=worst['variance'],
                     shot_variance=worst['shot_variance'],drift_variance=worst['drift_variance'],
                     resources=worst['resources'],local_success=bool(fit.success),
                     dense_variances=[r['variance'] for r in checks])
            rows.append(row)
            (OUT/f'drift_refined{args.tag}.json').write_text(json.dumps(dict(config=config,training_bs=bs,results=rows),indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
