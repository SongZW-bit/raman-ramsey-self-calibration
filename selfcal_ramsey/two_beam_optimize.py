"""Resource-matched pulse-family search in the two-beam noise model."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import differential_evolution,minimize
from drift_study import layout
from two_beam import evaluate,statistics,resources,control_size,atom_counts

OUT=Path(__file__).resolve().parent/'results'
BASE=[(.6,7.),(.6,12.),(.1,5.),(10.,120.),(.6,1.),(-np.pi,np.pi)]


def reshape(v,source,target):
    if len(v)>control_size(source):
        return reshape(v[:control_size(source)],source,target)+[v[control_size(source)]]
    if source==target:return list(v)
    if target=='flex':return reshape(v,source,'shape')+[0.]*4
    if source=='flex':return reshape(v[:12],'shape',target)
    if target=='shape':
        return list(v[:6])+([0.,0.,np.pi,np.pi] if source=='hyper' else [0.]*4)+list(v[6:])
    if source=='shape':return list(v[:6])+list(v[10:])
    return list(v)


def feasible(v,pattern,config):
    v=np.array(v,dtype=float)
    flags=layout(v,pattern,config)[0]
    nlong=sum(flags);nshort=len(flags)-nlong
    maximum=(config['time_cap']-len(flags)*(config['overhead']+v[0]+v[1])-nshort*v[2])/nlong
    v[3]=min(v[3],maximum-1e-8)
    return v


def polish(v,kind,pattern,config,bounds):
    active=list(np.linspace(-.6,.6,13));best=list(v)
    for cycle in range(2):
        free=dict(config,time_cap=1e8,exposure_cap=1e8)
        def cons(y):
            vals=evaluate(y[:-1],kind,pattern,free,active,True)['records']
            cost=resources(y[:-1],pattern,config)
            return np.r_[y[-1]-np.array([r['variance']/1e-8 for r in vals]),
                         (config['time_cap']-cost['elapsed'])/100.,
                         (config['exposure_cap']-cost['exposure'])/10.]
        initial=np.r_[best,evaluate(best,kind,pattern,config,active)/1e-8]
        fit=minimize(lambda y:y[-1],initial,method='SLSQP',bounds=bounds+[(.001,10000.)],
            constraints=[dict(type='ineq',fun=cons)],options=dict(maxiter=100,ftol=1e-8,eps=2e-6))
        if min(cons(fit.x))>=-1e-6 and fit.fun<initial[-1]:best=list(feasible(fit.x[:-1],pattern,config))
        dense=evaluate(best,kind,pattern,config,np.linspace(-.6,.6,81),True)['records']
        active=sorted(set(active+[r['b'] for r in sorted(dense,key=lambda z:z['variance'])[-4:]]))
    return best


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--readout',choices=['quad','two','two-group'],default='quad')
    parser.add_argument('--tau',type=float,default=300.)
    parser.add_argument('--rho',type=float,default=0.)
    parser.add_argument('--iterations',type=int,default=90)
    parser.add_argument('--seeds',type=int,nargs='+',default=[3121,7523])
    parser.add_argument('--families',nargs='+',default=['square','hyper','shape'])
    parser.add_argument('--counts',type=int,nargs='+',default=[8,10,12])
    parser.add_argument('--tag',default='')
    parser.add_argument('--allocate-atoms',action='store_true')
    args=parser.parse_args()
    source=json.loads((OUT/'drift_schedules.json').read_text())
    config=dict(source['config'],beam_correlation=args.rho,correlation_time=args.tau,
                b_width=.6,quadrature=args.readout=='quad',two_per_group=args.readout=='two-group')
    prior=[]
    for filename in ('drift_schedules.json','drift_refined_wide.json','drift_selected.json',
                     'two_beam_quad.json','two_beam_two.json','two_beam_two_flex.json',
                     'two_beam_quad_allocated.json','two_beam_two_allocated.json'):
        if not (OUT/filename).exists():continue
        prior+=json.loads((OUT/filename).read_text())['results']
    output=OUT/f'two_beam_{args.readout}{args.tag}.json'
    rows=json.loads(output.read_text())['results'] if output.exists() else []
    for kind in args.families:
        bounds=BASE+([(-np.pi,np.pi)]*4 if kind in ('shape','flex') else [])+[(-1.,1.),(-np.pi,np.pi)]
        if kind=='flex':bounds+=[(-2.5,2.5)]*4
        if args.allocate_atoms:bounds+=[(.05,.95)]
        for count in args.counts:
            pattern=f'distributed{count}'
            if any(r['family']==kind and r['pattern']==pattern for r in rows):continue
            candidates=[feasible(reshape(r['parameters'],r['family'],kind),pattern,config)
                for r in prior+rows if r['family']==kind or (kind in ('shape','flex') and r['family'] in ('square','hyper','shape'))]
            candidates=[np.r_[v[:control_size(kind)],v[control_size(kind)] if len(v)>control_size(kind) else count/config['shots']]
                        if args.allocate_atoms else v[:control_size(kind)] for v in candidates]
            candidates=[v for v in candidates if all(lo<=x<=hi for x,(lo,hi) in zip(v,bounds))
                and resources(v,pattern,config)['exposure']<=config['exposure_cap']]
            grid=np.linspace(-.6,.6,13)
            start=min(candidates,key=lambda v:evaluate(v,kind,pattern,config,grid))
            fits=[]
            for seed in args.seeds:
                fit=differential_evolution(lambda v:evaluate(v,kind,pattern,config,grid)/1e-8,
                    bounds,x0=start,seed=seed,maxiter=args.iterations,popsize=7,tol=2e-6,polish=False)
                candidates.append(fit.x)
                fits.append(dict(seed=seed,converged=bool(fit.success),objective=float(fit.fun)))
            best=min(candidates,key=lambda v:evaluate(v,kind,pattern,config,grid))
            best=polish(best,kind,pattern,config,bounds)
            dense=np.linspace(-.6,.6,81)
            checks=[statistics(best,kind,pattern,b,config,2) for b in dense]
            worst=max(range(len(checks)),key=lambda j:checks[j]['variance'])
            row=dict(family=kind,pattern=pattern,parameters=best,readout=args.readout,
                variance=checks[worst]['variance'],worst_b=float(dense[worst]),resources=checks[worst]['resources'],
                components={k:checks[worst][k] for k in ('shot_variance','common_variance','difference_variance')},
                atoms_per_shot=atom_counts(best,kind,layout(best,pattern,config)[0],config).tolist(),
                dense_variances=[z['variance'] for z in checks],search=fits)
            rows.append(row)
            output.write_text(json.dumps(dict(config=config,results=rows),indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k not in ('parameters','dense_variances')}),flush=True)


if __name__=='__main__':main()
