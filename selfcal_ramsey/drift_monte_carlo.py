"""Nonlinear OU trajectories and finite-atom inference, without true-b weights."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.interpolate import CubicSpline
from drift_study import layout
from drift_continuous import probability, pulse_samples, statistics

OUT=Path(__file__).resolve().parent/'results'


def lookup(design,config,grid):
    records=[statistics(design['parameters'],design['family'],design['pattern'],b,config,3) for b in grid]
    p=np.array([z['probabilities'] for z in records])
    a=np.array([z['jacobian'][:,0] for z in records])
    cov=np.array([z['covariance'] for z in records])
    precision=np.linalg.inv(cov)
    return p,a,precision,np.linalg.slogdet(cov)[1]


def fit_samples(y,grid,lookup_values):
    p,a,precision,logdet=lookup_values
    ca=np.einsum('gij,gj->gi',precision,a)
    denom=np.einsum('gi,gi->g',a,ca)
    out=[]
    for batch in np.array_split(y,max(1,len(y)//100)):
        residual=batch[:,None,:]-p[None,:,:]
        delta=np.einsum('ngi,gi->ng',residual,ca)/denom
        loss=np.einsum('ngi,gij,ngj->ng',residual,precision,residual)-delta**2*denom+logdet
        k=np.argmin(loss,axis=1)
        # Quadratic interpolation avoids quantizing the nuisance estimate.
        k=np.clip(k,1,len(grid)-2);ii=np.arange(len(batch))
        shift=(loss[ii,k-1]-loss[ii,k+1])/(2*(loss[ii,k-1]+loss[ii,k+1]-2*loss[ii,k])+1e-30)
        shift=np.clip(shift,-1.,1.)
        b=grid[k]+shift*(grid[1]-grid[0])
        ds=delta[ii,k]+shift*(delta[ii,k+1]-delta[ii,k-1])/2
        out.append(np.column_stack([ds,b]))
    return np.concatenate(out)


def generate(design,config,b,reps,rng,delta=0.,substeps=8):
    v=design['parameters'];kind=design['family'];pattern=design['pattern']
    flags,phases,_,_=layout(v,pattern,config)
    centers,_=pulse_samples(v,kind,flags,config,substeps)
    t=centers.ravel()
    noise=np.empty((reps,len(t)))
    noise[:,0]=rng.normal(0,config['sigma_intensity'],reps)
    for j in range(1,len(t)):
        corr=np.exp(-(t[j]-t[j-1])/config['correlation_time'])
        noise[:,j]=corr*noise[:,j-1]+config['sigma_intensity']*np.sqrt(1-corr*corr)*rng.normal(size=reps)
    p=probability(delta,b,noise.reshape(reps,*centers.shape),v,kind,flags,phases,config,substeps)
    return rng.binomial(config['atoms_per_shot'],p)/config['atoms_per_shot']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='drift_refined_quadrature.json')
    parser.add_argument('--reps',type=int,default=2000)
    parser.add_argument('--families',nargs='+',default=['square','hyper','shape'])
    parser.add_argument('--output',default='drift_monte_carlo.json')
    parser.add_argument('--by-pattern',action='store_true')
    parser.add_argument('--points',type=float,nargs='+',default=[-.6,-.3,0.,.3,.6])
    parser.add_argument('--delta',type=float,default=0.)
    args=parser.parse_args()
    data=json.loads((OUT/args.source).read_text())
    config=data['config']
    designs=[min((z for z in data['results'] if z['family']==kind),key=lambda z:z['variance']) for kind in args.families]
    if args.by_pattern:designs=[z for z in data['results'] if z['family'] in args.families]
    rows=[]
    grid=np.linspace(-.8,.8,401)
    for di,design in enumerate(designs):
        lu=lookup(design,config,grid)
        print('Lookup ready',design['family'],flush=True)
        for b in args.points:
            y=generate(design,config,b,args.reps,np.random.default_rng(1826+di*11+round((b+.6)*100)),delta=args.delta)
            fits=fit_samples(y,grid,lu)
            bias=float(fits[:,0].mean()-args.delta)
            err=(fits[:,0]-args.delta)**2
            row=dict(family=design['family'],pattern=design['pattern'],
                     design_label=design.get('label',design['family']+' '+str(design['pattern'])),b=b,reps=args.reps,
                     true_delta=args.delta,delta_bias=bias,delta_mse=float(err.mean()),
                     mse_standard_error=float(err.std(ddof=1)/np.sqrt(args.reps)),
                     nuisance_rmse=float(np.sqrt(np.mean((fits[:,1]-b)**2))),
                     nuisance_catastrophic_fraction=float(np.mean(abs(fits[:,1]-b)>.15)))
            rows.append(row)
            (OUT/args.output).write_text(json.dumps(dict(config=config,source=args.source,results=rows),indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
