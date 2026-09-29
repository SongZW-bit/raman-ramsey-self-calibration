"""Resolve OU intensity fluctuations inside each finite pulse."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from model import rotate
from drift_study import CONFIG, layout


def pulse_samples(v,kind,flags,config=CONFIG,substeps=4):
    t1,t2,ts,tl,power,offset=v[:6]
    dark=np.where(flags,tl,ts)
    lengths=config['overhead']+t1+dark+t2
    starts=np.r_[0.,np.cumsum(lengths)[:-1]]+config['overhead']
    n=3*substeps
    centers=np.concatenate([starts[:,None]+(np.arange(n)+.5)*t1/n,
                             starts[:,None]+t1+dark[:,None]+(np.arange(n)+.5)*t2/n],axis=1)
    if kind=='shape':angles=np.array([0.,v[6],v[7],v[8],v[9],0.])
    elif kind=='hyper':angles=np.array([0.,0.,0.,np.pi,np.pi,0.])
    else:angles=np.zeros(6)
    return centers,np.repeat(angles,substeps)


def probability(delta,b,noise,v,kind,flags,phases,config=CONFIG,substeps=4):
    noise=np.asarray(noise)
    if noise.ndim==2:noise=noise[None,:,:]
    ns,k,total=noise.shape
    extra=10 if kind=='shape' else 6
    freq_step=v[extra] if len(v)>extra else 0.
    short_phase=v[extra+1] if len(v)>extra+1 else 0.
    n=total//2
    _,angles=pulse_samples(v,kind,flags,config,substeps)
    delta=np.broadcast_to(np.asarray(delta).reshape(-1,1),(ns,k))
    b=np.broadcast_to(np.asarray(b).reshape(-1,1),(ns,k))
    r=np.zeros((ns,k,3));r[...,2]=1.
    dark=np.where(flags,v[3],v[2])[None,:]
    for s,angle in enumerate(angles):
        second=s>=n
        phi=angle+(phases[None,:]+v[5]+short_phase*(~flags)[None,:] if second else 0.)
        intensity=v[4]*(1+noise[:,:,s])
        h=np.stack([intensity*np.cos(phi),intensity*np.sin(phi),delta+b*intensity-freq_step],axis=-1)
        r=rotate(r,h,v[int(second)]/n)
        if s==n-1:
            h=np.zeros_like(r);h[...,2]=delta
            r=rotate(r,h,dark)
            r[...,:2]*=np.exp(-config['gamma']*dark)[...,None]
    return np.clip((1-r[...,2])/2,1e-10,1-1e-10)


def statistics(v,kind,pattern,b,config=CONFIG,substeps=4):
    flags,phases,_,cost=layout(v,pattern,config)
    centers,_=pulse_samples(v,kind,flags,config,substeps)
    k,n=centers.shape
    h=1e-5
    noise=np.zeros((1+2*n,k,n))
    for j in range(n):
        noise[1+2*j,:,j]=h;noise[2+2*j,:,j]=-h
    ps=probability(0.,b,noise,v,kind,flags,phases,config,substeps)
    jac=(ps[1::2]-ps[2::2]).T/(2*h)
    g=np.zeros((k,k*n))
    for j in range(k):g[j,j*n:(j+1)*n]=jac[j]
    t=centers.ravel()
    covnoise=config['sigma_intensity']**2*np.exp(-abs(t[:,None]-t[None,:])/config['correlation_time'])
    # Each shot responds only to its own pulse windows. Contract those blocks
    # directly instead of multiplying mostly-zero full response matrices.
    covdrift=np.einsum('ia,iajb,jb->ij',jac,covnoise.reshape(k,n,k,n),jac)
    p=ps[0]
    covariance=np.diag(p*(1-p)/config['atoms_per_shot'])+covdrift
    zeros=np.zeros((4,k,n))
    derivatives=probability([h,-h,0.,0.],[b,b,b+h,b-h],zeros,v,kind,flags,phases,config,substeps)
    a=np.stack([(derivatives[0]-derivatives[1])/(2*h),(derivatives[2]-derivatives[3])/(2*h)],axis=-1)
    sinva=cho_solve(cho_factor(covariance),a)
    info=a.T@sinva
    variance=1/(info[0,0]-info[0,1]**2/info[1,1])
    w=(sinva[:,0]-info[0,1]/info[1,1]*sinva[:,1])*variance
    return dict(variance=float(variance),shot_variance=float(np.sum(w*w*p*(1-p)/config['atoms_per_shot'])),
                drift_variance=float(w@covdrift@w),weights=w,covariance=covariance,
                intensity_jacobian=g,noise_covariance=covnoise,probabilities=p,
                jacobian=a,flags=flags,phases=phases,resources=cost)
