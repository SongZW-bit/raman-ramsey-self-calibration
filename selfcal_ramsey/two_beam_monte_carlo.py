"""Finite-atom nonlinear two-beam simulations with inferred light-shift mean."""
from pathlib import Path
import argparse
import json
import numpy as np
from drift_study import layout
from drift_monte_carlo import fit_samples
from two_beam import geometry,probability,statistics,atom_counts

OUT=Path(__file__).resolve().parent/'results'


def lookup(design,config,grid):
    statistics_function=statistics
    if config.get('quadratic_covariance',False):
        from two_beam_quadratic import statistics as statistics_function
    records=[statistics_function(design['parameters'],design['family'],design['pattern'],b,config,3,True) for b in grid]
    p=np.array([r['probabilities'] for r in records])
    a=np.array([r['jacobian'][:,0] for r in records])
    covariance=np.array([r['covariance'] for r in records])
    return p,a,np.linalg.inv(covariance),np.linalg.slogdet(covariance)[1]


def generate(design,config,b,reps,rng,delta,substeps=8):
    v=design['parameters'];kind=design['family'];pattern=design['pattern']
    flags,phases,_,_=layout(v,pattern,config)
    centers,_,_=geometry(v,kind,flags,config,substeps)
    times=centers.ravel();k,n=centers.shape
    noise=np.empty((reps,2,len(times)))
    rho=config['beam_correlation'];sigma=config['sigma_intensity']
    std=sigma*np.sqrt(np.array([(1+rho)/2,(1-rho)/2]))
    noise[:,:,0]=rng.normal(size=(reps,2))*std
    for j in range(1,len(times)):
        corr=np.exp(-(times[j]-times[j-1])/config['correlation_time'])
        noise[:,:,j]=corr*noise[:,:,j-1]+np.sqrt(1-corr*corr)*rng.normal(size=(reps,2))*std
    noise=noise.reshape(reps,2,k,n).transpose(0,2,1,3)
    p=probability(delta,b,noise,v,kind,flags,phases,config,substeps)
    counts=atom_counts(v,kind,flags,dict(config,integer_atoms=True))
    return rng.binomial(counts,p)/counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',nargs='+',default=['two_beam_quad.json','two_beam_two.json'])
    parser.add_argument('--families',nargs='+',default=['square','hyper','shape'])
    parser.add_argument('--reps',type=int,default=5000)
    parser.add_argument('--delta',type=float,default=.0005)
    parser.add_argument('--points',type=float,nargs='+',default=[-.6,-.45,-.3,-.15,0.,.15,.3,.45,.6])
    parser.add_argument('--output',default='two_beam_mc.json')
    parser.add_argument('--nonlinear',action='store_true')
    parser.add_argument('--starts',type=int,default=7)
    parser.add_argument('--b-fit-width',type=float,default=.8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--mean-correction',action='store_true')
    parser.add_argument('--quadratic-covariance',action='store_true')
    args=parser.parse_args()
    candidates=[]
    for filename in args.sources:
        source=json.loads((OUT/filename).read_text())
        for row in source['results']:
            config=dict(row.get('config',source['config']))
            if args.quadratic_covariance:config['quadratic_covariance']=True
            candidates.append((row,config))
    designs=[min((z for z in candidates if z[0]['family']==kind),key=lambda z:z[0]['variance']) for kind in args.families]
    rows=json.loads((OUT/args.output).read_text())['results'] if args.resume and (OUT/args.output).exists() else []
    for di,(design,config) in enumerate(designs):
        config=dict(config,integer_atoms=True)
        grid=np.linspace(-args.b_fit_width,args.b_fit_width,401)
        lu=lookup(design,config,grid)
        if args.nonlinear:
            from two_beam_inference import NonlinearEstimator
            estimator=NonlinearEstimator(design,config,grid,lu,mean_correction=args.mean_correction)
        print('Lookup ready '+design['family']+' '+design['readout'],flush=True)
        for b in args.points:
            if any(r['family']==design['family'] and r['readout']==design['readout'] and r['b']==b and
                   r['reps']==args.reps for r in rows):continue
            y=generate(design,config,b,args.reps,np.random.default_rng(6825+di*173+round((b+.6)*100)),args.delta)
            fits=estimator.fit(y,starts=args.starts)[0] if args.nonlinear else fit_samples(y,grid,lu)
            err=(fits[:,0]-args.delta)**2
            row=dict(family=design['family'],readout=design['readout'],pattern=design['pattern'],b=b,
                true_delta=args.delta,reps=args.reps,delta_bias=float(fits[:,0].mean()-args.delta),
                delta_mse=float(err.mean()),mse_standard_error=float(err.std(ddof=1)/np.sqrt(args.reps)),
                nuisance_rmse=float(np.sqrt(np.mean((fits[:,1]-b)**2))),
                nuisance_catastrophic_fraction=float(np.mean(abs(fits[:,1]-b)>.15)))
            rows.append(row)
            (OUT/args.output).write_text(json.dumps(dict(designs=[dict(design=d,config=c) for d,c in designs],
                inference='nonlinear_mean_gaussian_covariance' if args.nonlinear else 'linear_delta_mean',
                starts=args.starts if args.nonlinear else None,b_fit_width=args.b_fit_width,
                second_order_mean=args.mean_correction,
                results=rows),indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
