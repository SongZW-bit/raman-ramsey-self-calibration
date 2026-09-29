"""Convex minimax ensemble allocation for fixed pulse/sampling controls."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import minimize
from drift_study import layout
from two_beam import atom_counts, resources
from two_beam_quadratic import statistics

OUT=Path(__file__).resolve().parent/'results'


def moments(design,config,bs):
    v=design['parameters'];kind=design['family'];pattern=design['pattern']
    flags=layout(v,pattern,config)[0]
    counts=atom_counts(v,kind,flags,dict(config,integer_atoms=False))
    rows=[statistics(v,kind,pattern,b,config,3,True) for b in bs]
    kernels=[];projection=[]
    for row in rows:
        mu=row['probabilities']+row['mean_shift']
        # diag(C)=Kii+[mu(1-mu)-Kii]/Ni, with K independent of Ni.
        noise_diagonal=(np.diag(row['covariance'])-mu*(1-mu)/counts)/(1-1/counts)
        ai=mu*(1-mu)-noise_diagonal
        assert min(ai)>0
        kernels.append(row['covariance']-np.diag(ai/counts))
        projection.append(ai)
    return np.array(kernels),np.array(projection),np.array([r['jacobian'] for r in rows])


def objective(fractions,kernels,projection,jac,total):
    covariance=kernels.copy()
    ids=np.arange(len(fractions))
    covariance[:,ids,ids]+=projection/(total*fractions)
    solved=np.linalg.solve(covariance,jac)
    fisher=np.einsum('bni,bnj->bij',jac,solved)
    inv=np.linalg.inv(fisher)
    weights=np.einsum('bni,bi->bn',solved,inv[:,:,0])
    gradient=-projection*weights**2/(total*fractions**2)
    return inv[:,0,0],gradient,weights


def allocate(design,config,points=81):
    config=dict(config,integer_atoms=False)
    bs=np.linspace(-.6,.6,points)
    kernels,projection,jac=moments(design,config,bs)
    flags=layout(design['parameters'],design['pattern'],config)[0]
    counts=atom_counts(design['parameters'],design['family'],flags,config)
    total=float(counts.sum());initial=counts/total;k=len(counts)
    initial_values=objective(initial,kernels,projection,jac,total)[0]
    def cons(y):
        return y[-1]-objective(y[:-1],kernels,projection,jac,total)[0]/1e-8
    def derivative(y):
        return np.column_stack([-objective(y[:-1],kernels,projection,jac,total)[1]/1e-8,np.ones(points)])
    result=minimize(lambda y:y[-1],np.r_[initial,max(initial_values)/1e-8],
        jac=lambda y:np.r_[np.zeros(k),1.],method='SLSQP',
        bounds=[(100/total,1.)]*k+[(.001,1000.)],
        constraints=[dict(type='ineq',fun=cons,jac=derivative),
                     dict(type='eq',fun=lambda y:y[:-1].sum()-1.,jac=lambda y:np.r_[np.ones(k),0.])],
        options=dict(maxiter=300,ftol=1e-11))
    assert result.success and min(cons(result.x))>-1e-6
    weights=result.x[:-1]/result.x[:-1].sum()
    # Check the envelope derivative independently before recording a result.
    h=1e-6;direction=np.zeros(k);direction[:2]=[1.,-1.]
    numerical=(objective(weights+h*direction,kernels,projection,jac,total)[0]-
               objective(weights-h*direction,kernels,projection,jac,total)[0])/(2*h)
    analytic=objective(weights,kernels,projection,jac,total)[1]@direction
    error=float(np.max(abs(numerical-analytic))/max(np.max(abs(analytic)),1e-12))
    assert error<1e-4
    config.update(atom_weights=weights.tolist(),total_atoms=int(total),integer_atoms=True)
    finite=[statistics(design['parameters'],design['family'],design['pattern'],b,config,3) for b in bs]
    worst=max(finite,key=lambda r:r['variance'])
    row=dict(design,config=config,variance=worst['variance'],readout=design['readout']+'-atom-opt',
             atoms_per_shot=atom_counts(design['parameters'],design['family'],flags,config).tolist(),
             resources=resources(design['parameters'],design['pattern'],config),
             allocation_audit=dict(success=bool(result.success),iterations=int(result.nit),
                original_variance=float(max(initial_values)),continuous_variance=float(result.fun*1e-8),
                derivative_relative_error=error,minimum_atoms=100,
                scope='Convex fixed-control, finite-grid quadratic-noise allocation; not an exact-likelihood bound.'))
    row['components']={key:worst[key] for key in ('shot_variance','common_variance','difference_variance','quadratic_variance')}
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_ratio_smooth.json')
    parser.add_argument('--output',default='two_beam_atom_design.json')
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    source=json.loads((OUT/args.source).read_text())
    chosen=source['results'] if args.all else [min((r for r in source['results'] if r['family']==kind),key=lambda r:r['variance']) for kind in ('hyper','shape')]
    rows=[]
    for design in chosen:
        row=allocate(design,design.get('config',source['config']))
        rows.append(row)
        print(json.dumps(dict(family=row['family'],readout=row['readout'],variance=row['variance'],audit=row['allocation_audit'])),flush=True)
        (OUT/args.output).write_text(json.dumps(dict(config=source['config'],results=rows),indent=2),encoding='utf-8')


if __name__=='__main__':main()
