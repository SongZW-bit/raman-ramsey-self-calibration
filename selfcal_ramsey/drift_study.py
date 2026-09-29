"""Correlated intensity noise in serial ensemble Ramsey measurements.

All shots, including short calibration shots, are counted. The objective is
the variance of the best locally unbiased linear estimator with a static
unknown light-shift coefficient. It includes shot noise and OU covariance
at the two pulse centers of every interrogation.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution
from model import rotate

OUT = Path(__file__).resolve().parent/'results'
CONFIG = dict(shots=16, atoms_per_shot=10000, time_cap=480.,
              exposure_cap=160., overhead=2., gamma=.01,
              sigma_intensity=.03, correlation_time=30.)


def sequence(x, v, kind, long_flags, phases, gamma=.01):
    x=np.atleast_2d(x)
    t1,t2,ts,tl,power,offset=v[:6]
    extra=10 if kind=='shape' else 6
    freq_step=v[extra] if len(v)>extra else 0.
    short_phase=v[extra+1] if len(v)>extra+1 else 0.
    if kind=='shape':
        first=(0.,v[6],v[7]); second=(v[8],v[9],0.)
    elif kind=='hyper':
        first=(0.,0.,0.); second=(np.pi,np.pi,0.)
    else:
        first=second=(0.,0.,0.)
    dark=np.where(long_flags,tl,ts)[None,:]
    dims=(len(x),len(long_flags))
    delta=np.broadcast_to(x[:,0,None],dims)
    r=np.zeros(dims+(3,));r[...,2]=1.
    for pulse,(duration,angles) in enumerate(((t1,first),(t2,second))):
        intensity=np.broadcast_to(power*(1+x[:,2+pulse,None]),dims)
        det=delta+x[:,1,None]*intensity-freq_step
        for angle in angles:
            phi=angle+(phases[None,:]+offset+short_phase*(~long_flags)[None,:] if pulse else 0.)
            omega=np.stack([intensity*np.cos(phi),intensity*np.sin(phi),det],axis=-1)
            r=rotate(r,omega,duration/3)
        if pulse==0:
            omega=np.zeros_like(r);omega[...,2]=delta
            r=rotate(r,omega,dark)
            r[...,:2]*=np.exp(-gamma*dark)[...,None]
    return np.clip((1-r[...,2])/2,1e-10,1-1e-10)


def layout(v, pattern, config=CONFIG):
    k=config['shots'];short=k//2
    if isinstance(pattern,dict):flags=np.asarray(pattern['long_flags'],dtype=bool)
    elif pattern=='alternating':flags=np.arange(k)%2==1
    elif pattern=='blocks':flags=np.arange(k)>=short
    elif pattern=='abba':flags=np.tile([False,True,True,False],k//4)
    elif pattern=='pairs':flags=np.tile([False,False,True,True],k//4)
    elif pattern.startswith('distributed'):
        nshort=int(pattern.removeprefix('distributed'))
        flags=np.ones(k,dtype=bool)
        flags[np.floor(np.arange(nshort)*k/nshort).astype(int)]=False
    else:flags=np.asarray(pattern,dtype=bool)
    t1,t2,ts,tl,power=v[:5]
    dark=np.where(flags,tl,ts)
    duration=config['overhead']+t1+dark+t2
    starts=np.r_[0.,np.cumsum(duration)[:-1]]+config['overhead']
    centers=np.column_stack([starts+t1/2,starts+t1+dark+t2/2]).ravel()
    phases=np.where(np.arange(k)%2,np.pi/2,-np.pi/2)
    if config.get('quadrature',False):
        for flag in (False,True):
            mask=np.flatnonzero(flags==flag)
            phases[mask]=np.resize([0.,np.pi/2,np.pi,3*np.pi/2],len(mask))
    elif config.get('two_per_group',False):
        for flag in (False,True):
            mask=np.flatnonzero(flags==flag)
            phases[mask]=np.resize([-np.pi/2,np.pi/2],len(mask))
    if isinstance(pattern,dict):phases=np.asarray(pattern['phases'])
    return flags,phases,centers,dict(atoms=k*config['atoms_per_shot'],
                                   elapsed=float(duration.sum()),
                                   exposure=float(k*power*(t1+t2)),peak=float(power))


def evaluate(v,kind='square',pattern='alternating',config=CONFIG,bs=(.2,.3,.4),details=False):
    flags,phases,centers,cost=layout(v,pattern,config)
    if cost['elapsed']>config['time_cap']*(1+1e-10) or cost['exposure']>config['exposure_cap']*(1+1e-10):
        excess=max(cost['elapsed']/config['time_cap'],cost['exposure']/config['exposure_cap'])
        return 1e-4*(excess**2) if not details else dict(invalid=True,resources=cost)
    xs=np.array([[0.,b,0.,0.] for b in bs])
    h=1e-5
    samples=np.concatenate([xs]+[xs+sgn*h*np.eye(4)[j] for j in range(4) for sgn in (1,-1)])
    allp=sequence(samples,v,kind,flags,phases,config['gamma']).reshape(9,len(bs),-1)
    p=allp[0]
    jac=np.stack([(allp[1+2*j]-allp[2+2*j])/(2*h) for j in range(4)],axis=-1)
    kernel=config['sigma_intensity']**2*np.exp(-abs(centers[:,None]-centers[None,:])/config['correlation_time'])
    if config.get('noise_model')=='shot':
        shot_ids=np.repeat(np.arange(len(flags)),2)
        kernel=config['sigma_intensity']**2*(shot_ids[:,None]==shot_ids[None,:])
    records=[]
    for bi,b in enumerate(bs):
        g=np.zeros((len(flags),len(centers)))
        g[np.arange(len(flags)),2*np.arange(len(flags))]=jac[bi,:,2]
        g[np.arange(len(flags)),2*np.arange(len(flags))+1]=jac[bi,:,3]
        covariance=np.diag(p[bi]*(1-p[bi])/config['atoms_per_shot'])+g@kernel@g.T
        a=jac[bi,:,:2]
        solved=cho_solve(cho_factor(covariance),a)
        info=a.T@solved
        effective=info[0,0]-info[0,1]**2/max(info[1,1],1e-25)
        variance=1/max(effective,1e-25)
        w=(solved[:,0]-info[0,1]/max(info[1,1],1e-25)*solved[:,1])*variance
        shot=float(np.sum(w*w*p[bi]*(1-p[bi])/config['atoms_per_shot']))
        records.append(dict(b=float(b),variance=float(variance),shot_variance=shot,
                            drift_variance=float(w@(g@kernel@g.T)@w),
                            weights=w.tolist(),intensity_jacobian=g.tolist(),
                            probabilities=p[bi].tolist(),target_jacobian=a[:,0].tolist(),
                            nuisance_jacobian=a[:,1].tolist()))
    worst=max(r['variance'] for r in records)
    return dict(variance=worst,records=records,resources=cost,parameters=list(v),
                family=kind,pattern=pattern) if details else worst


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=90)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--overhead',type=float,default=2.)
    parser.add_argument('--corr',type=float,default=30.)
    parser.add_argument('--sigma',type=float,default=.03)
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    CONFIG['time_cap']+=CONFIG['shots']*(args.overhead-CONFIG['overhead'])
    CONFIG['overhead']=args.overhead
    CONFIG['correlation_time']=args.corr
    CONFIG['sigma_intensity']=args.sigma
    bounds=[(.6,5.),(.6,7.),(.1,5.),(10.,50.),(.6,1.),(-np.pi,np.pi)]
    rows=[]
    for kind in ('square','hyper','shape'):
        for pattern in ('alternating','abba'):
            b=bounds+([(-np.pi,np.pi)]*4 if kind=='shape' else [])
            x0=None
            if kind=='shape':
                base=min((z for z in rows if z['pattern']==pattern),key=lambda z:z['variance'])
                x0=base['parameters']+([0.,0.,np.pi,np.pi] if base['family']=='hyper' else [0.]*4)
            started=time.monotonic()
            fit=differential_evolution(lambda x:evaluate(x,kind,pattern)/1e-8,b,
                                       x0=x0,seed=args.seed,maxiter=args.iterations,
                                       popsize=7,tol=1e-6,polish=True)
            best=fit.x
            if x0 is not None and evaluate(x0,kind,pattern)<evaluate(best,kind,pattern):best=x0
            row=evaluate(best,kind,pattern,details=True)
            row.update(seconds=time.monotonic()-started,optimizer_success=bool(fit.success))
            rows.append(row)
            OUT.mkdir(exist_ok=True)
            (OUT/f'drift_{args.tag}{args.seed}.json').write_text(json.dumps(dict(config=CONFIG,results=rows),indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
