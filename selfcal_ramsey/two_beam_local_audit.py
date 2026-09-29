"""Audit nonlinear noise against the optimized local linear variance.

The added quadratic covariance is exact for a quadratic Taylor surrogate,
not the full fourth-order expansion of the physical covariance.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from drift_study import layout
from two_beam import probability, statistics, noise_kernels, atom_counts
from two_beam_monte_carlo import generate

OUT=Path(__file__).resolve().parent/'results'


def quadratic_terms(design,config,b,step=1e-4):
    v=design['parameters'];kind=design['family']
    flags,phases,times,_=layout(v,design['pattern'],config);k=len(flags)
    points=[np.zeros(4)];pairs=[]
    for a in range(4):
        points.extend([step*np.eye(4)[a],-step*np.eye(4)[a]])
    for a in range(4):
        for c in range(a+1,4):
            pairs.append((a,c,len(points)))
            points.extend([step*(sa*np.eye(4)[a]+sc*np.eye(4)[c])
                           for sa,sc in [(1,1),(1,-1),(-1,1),(-1,-1)]])
    points=np.asarray(points);noise=np.zeros((len(points),k,2,6))
    for a in range(4):noise[:,:,a//2,3*(a%2):3*(a%2)+3]=points[:,a,None,None]
    p=probability(0.,b,noise,v,kind,flags,phases,config)
    hessian=np.zeros((k,4,4))
    for a in range(4):hessian[:,a,a]=(p[1+2*a]+p[2+2*a]-2*p[0])/step**2
    for a,c,j in pairs:
        hessian[:,a,c]=hessian[:,c,a]=(p[j]-p[j+1]-p[j+2]+p[j+3])/(4*step**2)
    kernels=noise_kernels(times,config);cross=np.zeros((k,k,4,4))
    for channel in range(2):
        cross[:,:,2*channel:2*channel+2,2*channel:2*channel+2]=kernels[channel].reshape(k,2,k,2).transpose(0,2,1,3)
    shift=.5*np.einsum('iab,iab->i',hessian,cross[np.arange(k),np.arange(k)])
    quadratic=.5*np.einsum('iab,ijbc,jcd,ijad->ij',hessian,cross,hessian,cross)
    return shift,quadratic


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_joint_coded.json')
    parser.add_argument('--reps',type=int,default=12000)
    parser.add_argument('--points',nargs='+',type=float,default=[-.6,0.,.6])
    parser.add_argument('--output',default='two_beam_local_audit.json')
    args=parser.parse_args();source=json.loads((OUT/args.source).read_text())
    config=dict(source['config'],integer_atoms=True);rows=[]
    for kind in ('hyper','shape'):
        design=min((d for d in source['results'] if d['family']==kind),key=lambda d:d['variance'])
        for b in args.points:
            local=statistics(design['parameters'],kind,design['pattern'],b,config,3,True)
            shift,quadratic=quadratic_terms(design,config,b)
            half_shift,half_quad=quadratic_terms(design,config,b,5e-5)
            assert np.max(abs(shift-half_shift))<1e-7
            assert np.max(abs(quadratic-half_quad))<1e-8
            assert np.linalg.eigvalsh(quadratic).min()>-1e-12
            weights=local['weights'];rng=np.random.default_rng(29971+round((b+.6)*100))
            projections=[]
            for start in range(0,args.reps,1000):
                data=generate(design,config,b,min(1000,args.reps-start),rng,0.)
                projections.extend((data-local['probabilities']-shift)@weights)
            projections=np.asarray(projections)
            centered=(projections-projections.mean())**2
            counts=atom_counts(design['parameters'],kind,layout(design['parameters'],design['pattern'],config)[0],config)
            p=local['probabilities'];mu=p+shift
            noise_var=np.diag(local['covariance'])-p*(1-p)/counts+np.diag(quadratic)
            shot_correction=(mu*(1-mu)-noise_var-p*(1-p))/counts
            quadratic_added=float(weights@quadratic@weights)
            corrected=local['variance']+quadratic_added+float(np.sum(weights**2*shot_correction))
            row=dict(family=kind,b=b,reps=args.reps,linear_variance=local['variance'],
                quadratic_added_variance=quadratic_added,quadratic_surrogate_variance=corrected,
                empirical_oracle_linear_variance=float(np.var(projections,ddof=1)),
                empirical_variance_standard_error=float(centered.std(ddof=1)/np.sqrt(args.reps)),
                residual_bias=float(projections.mean()),
                scope='Oracle local weights at true b; diagnostic only, not an attainable unknown-b estimator.')
            rows.append(row);print(json.dumps(row),flush=True)
            (OUT/args.output).write_text(json.dumps(dict(source=args.source,config=config,results=rows),indent=2),encoding='utf-8')


if __name__=='__main__':main()
