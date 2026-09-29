"""Optimize the no-drift design, then test it under correlated drift."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import differential_evolution
from drift_study import evaluate
from drift_continuous import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['absent','quasistatic','shot'],default='absent')
    parser.add_argument('--seeds',type=int,nargs='+',default=[447,883])
    args=parser.parse_args()
    src=json.loads((OUT/'drift_schedules.json').read_text())
    config=src['config'];static=dict(config,sigma_intensity=0.)
    if args.mode=='quasistatic':static=dict(config,correlation_time=1e9)
    if args.mode=='shot':static=dict(config,noise_model='shot')
    bounds=[(.6,5.),(.6,7.),(.1,5.),(10.,120.),(.6,1.),(-np.pi,np.pi),(-1.,1.),(-np.pi,np.pi)]
    bs=np.linspace(-.6,.6,13)
    rows=[]
    for nshort in (8,10,12):
        pattern=f'distributed{nshort}'
        prior=next(z for z in src['results'] if z['family']=='square' and z['pattern']==pattern)
        fits=[differential_evolution(lambda v:evaluate(v,'square',pattern,static,bs)/1e-8,bounds,
                                    seed=seed,maxiter=250,popsize=7,x0=prior['parameters'],tol=1e-6) for seed in args.seeds]
        fit=min(fits,key=lambda z:z.fun)
        v=fit.x.tolist()
        predicted=evaluate(v,'square',pattern,static,bs,details=True)
        checks=[statistics(v,'square',pattern,b,config,6) for b in np.linspace(-.6,.6,61)]
        row=dict(family='square',pattern=pattern,parameters=v,static_variance=predicted['variance'],
                 variance=max(z['variance'] for z in checks),resources=checks[0]['resources'],
                 optimizer_success=bool(fit.success))
        rows.append(row)
        (OUT/f'drift_static_{args.mode}.json').write_text(json.dumps(dict(config=config,design_config=static,results=rows),indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
