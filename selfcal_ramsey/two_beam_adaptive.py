"""One-step posterior-risk control selection with counted calibration data.

Fixed physical timings permit a single OU trajectory through both stages.
The first four observations choose a frequency step for subsequent pulses.
Selection uses neither the true b nor the latent intensity trajectory.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import logsumexp
from drift_study import layout
from two_beam import probability, geometry, atom_counts, resources
from two_beam_quadratic import evaluate
from two_beam_monte_carlo import lookup
from two_beam_inference import NonlinearEstimator

OUT=Path(__file__).resolve().parent/'results'


def prepare(source,family,ncal=4):
    base=min((r for r in source['results'] if r['family']==family),key=lambda r:r['variance'])
    config=dict(base.get('config',source['config']),integer_atoms=True)
    flags=np.array(base['pattern']['long_flags']);phases=np.array(base['pattern']['phases'])
    selected=np.flatnonzero(~flags)[:ncal]
    order=np.r_[selected,np.array([i for i in range(len(flags)) if i not in selected])]
    assert len(selected)==ncal
    config['atom_weights']=np.array(base['atoms_per_shot'])[order].tolist()
    pattern=dict(long_flags=flags[order].tolist(),phases=phases[order].tolist())
    design=dict(base,pattern=pattern,readout='posterior-risk-prefix')
    return design,config


def branch_config(config,offset,ncal):
    control=np.atleast_1d(offset)
    assert len(control) in (1,5)
    updated=dict(config,frequency_offsets=[0.]*ncal+[float(control[0])]*(config['shots']-ncal))
    if len(control)==5:
        phases=np.zeros((config['shots'],6))
        phases[ncal:,1:5]=control[1:]
        updated['segment_phase_offsets']=phases.tolist()
    return updated


def bank(design,config,ncal):
    offsets=[0.]
    for center in np.linspace(-.6,.6,7):
        bs=np.clip(center+np.array([-.08,0.,.08]),-.6,.6)
        def risk(offset):
            return evaluate(design['parameters'],design['family'],design['pattern'],
                            branch_config(config,offset,ncal),bs)/1e-8
        grid=np.linspace(-.9,.9,25);losses=np.array([risk(x) for x in grid]);i=int(np.argmin(losses))
        fit=minimize_scalar(risk,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),
                            method='bounded',options=dict(xatol=1e-5))
        offset=float(fit.x) if fit.fun<losses[i] else float(grid[i])
        if min(abs(offset-x) for x in offsets)>.01:offsets.append(offset)
    return offsets


def phase_bank(design,config,ncal):
    scalar=bank(design,config,ncal);controls=[]
    domains=[np.linspace(-.6,.6,25)]+[np.clip(center+np.array([-.08,0.,.08]),-.6,.6) for center in np.linspace(-.6,.6,7)]
    for branch,bs in enumerate(domains):
        def losses(x):
            rows=evaluate(design['parameters'],design['family'],design['pattern'],
                          branch_config(config,x,ncal),bs,True)['records']
            return np.array([r['variance']/1e-8 for r in rows])
        start=min((np.r_[s,np.zeros(4)] for s in scalar),key=lambda x:max(losses(x)))
        best=start.copy();best_value=max(losses(best));history=[]
        for attempt in range(2):
            initial=best.copy()
            if attempt:initial[1:]+=np.random.default_rng(3912+branch).uniform(-.2,.2,4)
            fit=minimize(lambda y:y[-1],np.r_[initial,max(losses(initial))],method='SLSQP',
                         bounds=[(-.9,.9)]+[(-np.pi,np.pi)]*4+[(.001,1000.)],
                         constraints=[dict(type='ineq',fun=lambda y:y[-1]-losses(y[:-1]))],
                         options=dict(maxiter=100,ftol=2e-9,eps=2e-6))
            value=max(losses(fit.x[:-1]))
            if value<best_value:best=fit.x[:-1];best_value=value
            history.append(dict(success=bool(fit.success),message=str(fit.message),risk=float(value*1e-8)))
        controls.append(best.tolist())
        print(json.dumps(dict(phase_bank_branch=branch,risk=best_value*1e-8,controls=best.tolist(),history=history)),flush=True)
    return controls


def posterior(y,means,covariance,delta_jac,delta_grid):
    precision=np.linalg.inv(covariance)
    logdet=np.linalg.slogdet(covariance)[1]
    residual=y[:,None,None,:]-means[None,:,None,:]-delta_grid[None,None,:,None]*delta_jac[None,:,None,:]
    loss=np.einsum('nbdi,bij,nbdj->nbd',residual,precision,residual)+logdet[None,:,None]
    log_probability=logsumexp(-.5*loss,axis=2)
    return np.exp(log_probability-logsumexp(log_probability,axis=1)[:,None])


def trajectories(design,config,b,reps,rng,delta,substeps=8):
    v=design['parameters'];flags,phases,_,_=layout(v,design['pattern'],config)
    centers=geometry(v,design['family'],flags,config,substeps)[0]
    times=centers.ravel();noise=np.empty((reps,2,len(times)))
    std=config['sigma_intensity']*np.sqrt(np.array([1+config['beam_correlation'],1-config['beam_correlation']])/2)
    noise[:,:,0]=rng.normal(size=(reps,2))*std
    for j in range(1,len(times)):
        r=np.exp(-(times[j]-times[j-1])/config['correlation_time'])
        noise[:,:,j]=r*noise[:,:,j-1]+np.sqrt(1-r*r)*rng.normal(size=(reps,2))*std
    noise=noise.reshape(reps,2,len(flags),-1).transpose(0,2,1,3)
    return noise,flags,phases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_count_atoms.json')
    parser.add_argument('--family',default='hyper')
    parser.add_argument('--reps',type=int,default=1000)
    parser.add_argument('--output',default='two_beam_adaptive_pilot.json')
    parser.add_argument('--calibration-shots',type=int,default=4)
    parser.add_argument('--points',type=float,nargs='+',default=[-.6,-.3,0.,.3,.6])
    parser.add_argument('--delta',type=float,default=.0005)
    parser.add_argument('--seed',type=int,default=8501)
    parser.add_argument('--bank-source')
    parser.add_argument('--phase-bank',action='store_true')
    parser.add_argument('--modes',nargs='+',choices=['adaptive','fixed'],default=['adaptive','fixed'])
    args=parser.parse_args()
    source=json.loads((OUT/args.source).read_text())
    ncal=args.calibration_shots
    design,config=prepare(source,args.family,ncal)
    if args.bank_source:
        saved=json.loads((OUT/args.bank_source).read_text())
        assert saved['config']==config and saved['design']['parameters']==design['parameters']
        assert saved['design']['pattern']==design['pattern'] and saved['calibration_shots']==ncal
        offsets=saved['offsets']
    else:offsets=phase_bank(design,config,ncal) if args.phase_bank else bank(design,config,ncal)
    configs=[branch_config(config,offset,ncal) for offset in offsets]
    grid=np.linspace(-.8,.8,161)
    estimators=[];risks=[]
    for ci,cfg in enumerate(configs):
        lu=lookup(design,cfg,grid)
        estimator=NonlinearEstimator(design,cfg,grid,lu,mean_correction=True)
        estimators.append(estimator)
        risks.append([evaluate(design['parameters'],design['family'],design['pattern'],cfg,[b]) for b in grid])
        print(json.dumps(dict(branch=ci,offset=offsets[ci],maximum_risk=max(risks[-1]))),flush=True)
    risks=np.array(risks)
    central=abs(grid)<=.60001
    fixed=int(np.argmin(np.max(risks[:,central],axis=1)))
    first=estimators[0]
    # Uniform design prior over b and delta; b is marginalized over finite grids.
    means=first.p_spline(grid)[:,:ncal]
    covariance=np.linalg.inv(first.lookup[2])[:,:ncal,:ncal]
    delta_jac=first.a_spline(grid)[:,:ncal]
    for estimator in estimators[1:]:
        assert np.max(abs(estimator.p_spline(grid)[:,:ncal]-means))<1e-10
        assert np.max(abs(np.linalg.inv(estimator.lookup[2])[:,:ncal,:ncal]-covariance))<1e-10
    cost=resources(design['parameters'],design['pattern'],config)
    assert cost['atoms']==160000 and cost['elapsed']<=config['time_cap']*(1+1e-10)
    for cfg in configs:assert resources(design['parameters'],design['pattern'],cfg)==cost
    rows=[];paired=[];saved_errors={}
    result=dict(design=design,config=config,offsets=offsets,fixed_branch=fixed,resources=cost,
                control_bank='frequency_and_segment_phase' if np.ndim(offsets)==2 else 'frequency_only',
                calibration_shots=ncal,design_b_grid=grid.tolist(),design_risks=risks.tolist(),
                seed=args.seed,true_delta=args.delta,paired_comparisons=paired,
                prefix_and_resource_checks=True,
                scope='One-step Bayesian risk selection. Full OU trajectories; conditional design likelihood approximated by fixed-branch Gaussian density. Frequency switching adds no modeled settling cost.',results=rows)
    for b in args.points:
        rng=np.random.default_rng(args.seed+round((b+.6)*10000))
        noise,flags,phases=trajectories(design,config,b,args.reps,rng,args.delta)
        v=design['parameters'];kind=design['family']
        counts=atom_counts(v,kind,flags,config)
        p0=probability(args.delta,b,noise,v,kind,flags,phases,configs[0],8)
        ycal=rng.binomial(counts[:ncal],p0[:,:ncal])/counts[:ncal]
        choice=np.empty(args.reps,int)
        for begin in range(0,args.reps,400):
            post=posterior(ycal[begin:begin+400],means,covariance,delta_jac,np.linspace(-.004,.004,17))
            choice[begin:begin+400]=np.argmin(post@risks.T,axis=1)
        errors_by_mode={}
        for mode in args.modes:
            selected=choice if mode=='adaptive' else np.full(args.reps,fixed)
            fits=np.empty((args.reps,2))
            for branch in np.unique(selected):
                ids=np.flatnonzero(selected==branch)
                p=probability(args.delta,b,noise[ids],v,kind,flags,phases,configs[branch],8)
                assert np.max(abs(p[:,:ncal]-p0[ids,:ncal]))<1e-12
                y=np.column_stack([ycal[ids],rng.binomial(counts[ncal:],p[:,ncal:])/counts[ncal:]])
                fits[ids]=estimators[branch].fit(y)[0]
            errors=(fits[:,0]-args.delta)**2
            errors_by_mode[mode]=errors
            saved_errors[f'{mode}_{b}']=errors
            row=dict(mode=mode,b=b,reps=args.reps,mse=float(errors.mean()),
                     mse_standard_error=float(errors.std(ddof=1)/np.sqrt(args.reps)),
                     bias=float(fits[:,0].mean()-args.delta),
                     nuisance_catastrophic_fraction=float(np.mean(abs(fits[:,1]-b)>.15)),
                     branch_counts=np.bincount(selected,minlength=len(offsets)).tolist())
            rows.append(row);print(json.dumps(row),flush=True)
            (OUT/args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
        if 'fixed' in errors_by_mode and 'adaptive' in errors_by_mode:
            difference=errors_by_mode['fixed']-errors_by_mode['adaptive']
            paired.append(dict(b=b,mean_mse_difference=float(difference.mean()),
                               difference_standard_error=float(difference.std(ddof=1)/np.sqrt(args.reps))))
        (OUT/args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
        np.savez_compressed(OUT/Path(args.output).with_suffix('.npz'),**saved_errors)


if __name__=='__main__':main()
