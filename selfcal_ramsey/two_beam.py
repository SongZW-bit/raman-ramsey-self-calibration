"""Effective Lambda Raman model with distinct correlated beam intensities.

I1/I1_nom=1+c+d, I2/I2_nom=1+c-d. For a single excited state and
Omega_eff_nom=1, a=sqrt(1+b^2) fixes the sum of the two AC Stark shifts.
Thus Omega=u*sqrt((1+c)^2-d^2), shift=u*(b*(1+c)-a*d).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from model import rotate
from drift_study import layout
from drift_continuous import pulse_samples


def control_size(kind):
    return {'square':8,'hyper':8,'shape':12,'flex':16}[kind]


def atom_counts(v,kind,flags,config):
    total=config.get('total_atoms',config['shots']*config['atoms_per_shot'])
    index=control_size(kind)
    fraction=v[index] if len(v)>index else np.mean(~flags)
    counts=np.where(flags,total*(1-fraction)/sum(flags),total*fraction/sum(~flags))
    if 'atom_weights' in config:
        weights=np.asarray(config['atom_weights'],float)
        if weights.shape!=flags.shape or np.any(weights<=0):
            raise ValueError('Atom weights must be positive and match the interrogation count')
        counts=total*weights/weights.sum()
    if config.get('integer_atoms',False):
        rounded=np.floor(counts).astype(int)
        order=np.argsort(-(counts-rounded),kind='stable')
        rounded[order[:int(total-rounded.sum())]]+=1
        return rounded
    return counts


def geometry(v,kind,flags,config,substeps):
    if kind!='flex':
        centers,angles=pulse_samples(v,kind,flags,config,substeps)
        durations=np.repeat([v[0]/(3*substeps),v[1]/(3*substeps)],3*substeps)
        return centers,angles,durations
    logits=np.array([[v[12],v[13],0.],[v[14],v[15],0.]])
    weights=np.exp(logits-logits.max(axis=1,keepdims=True))
    fractions=.03+.91*weights/weights.sum(axis=1,keepdims=True)
    segments=fractions*np.array(v[:2])[:,None]
    durations=np.repeat(segments.ravel()/substeps,substeps)
    angles=np.repeat([0.,v[6],v[7],v[8],v[9],0.],substeps)
    dark=np.where(flags,v[3],v[2])
    shot_lengths=config['overhead']+v[0]+dark+v[1]
    starts=np.r_[0.,np.cumsum(shot_lengths)[:-1]]+config['overhead']
    local=np.cumsum(durations)-durations/2
    centers=starts[:,None]+local[None,:]
    centers[:,3*substeps:]+=dark[:,None]
    return centers,angles,durations


def resources(v,pattern,config):
    flags,_,_,cost=layout(v,pattern,config)
    width=config.get('b_width',.6);factor=np.sqrt(1+width**2)
    q=ratio_matrix(v,flags,config)
    durations=np.array(v[:2])[None,:]
    exposure=v[4]*(factor*np.sum(np.cosh(q)*durations)+width*abs(np.sum(np.sinh(q)*durations)))
    peak=v[4]*np.max(factor*np.cosh(q)+width*abs(np.sinh(q)))
    return dict(cost,atoms=config.get('total_atoms',cost['atoms']),exposure=float(exposure),peak=float(peak),
                nominal_coupling_exposure=cost['exposure'],max_stark_sum=factor)


def ratio_matrix(v,flags,config):
    """Known log-ratio offsets, constant within each pulse."""
    q=np.zeros((len(flags),2))
    if config.get('ratio_modulation',False):
        values=np.asarray(v[-4:]).reshape(2,2)
        for flag in (False,True):
            ids=np.flatnonzero(flags==flag)
            signs=np.where(np.arange(len(ids))%2,-1.,1.) if config.get('ratio_alternating',False) else np.ones(len(ids))
            q[ids]=signs[:,None]*values[int(flag)]
    return q


def probability(delta,b,noise,v,kind,flags,phases,config,substeps=1):
    noise=np.asarray(noise)
    ns,k,channels,total=noise.shape
    assert channels==2 and total==6*substeps
    extra=10 if kind in ('shape','flex') else 6
    frequency=v[extra]+np.asarray(config.get('frequency_offsets',0.));short_phase=v[extra+1]
    _,angles,durations=geometry(v,kind,flags,config,substeps)
    delta=np.broadcast_to(np.asarray(delta).reshape(-1,1),(ns,k))
    b=np.broadcast_to(np.asarray(b).reshape(-1,1),(ns,k))
    a=np.sqrt(1+b*b)
    q=ratio_matrix(v,flags,config)
    shifted_b=b[:,:,None]*np.cosh(q)[None,:,:]+a[:,:,None]*np.sinh(q)[None,:,:]
    shifted_a=a[:,:,None]*np.cosh(q)[None,:,:]+b[:,:,None]*np.sinh(q)[None,:,:]
    r=np.zeros((ns,k,3));r[...,2]=1.
    segment_phases=np.asarray(config.get('segment_phase_offsets',np.zeros((k,6))))
    if segment_phases.shape!=(k,6):raise ValueError('Expected six phase offsets per interrogation')
    dark=np.where(flags,v[3],v[2])[None,:]
    n=total//2
    for s,angle in enumerate(angles):
        second=s>=n
        phase=angle+segment_phases[None,:,s//substeps]+(phases[None,:]+v[5]+short_phase*(~flags)[None,:] if second else 0.)
        common=noise[:,:,0,s];difference=noise[:,:,1,s]
        product=(1+common)**2-difference**2
        if np.any(product<0):raise ValueError('Negative optical intensity in the Gaussian noise sample')
        omega=v[4]*np.sqrt(product)
        shift=v[4]*(shifted_b[:,:,int(second)]*(1+common)-shifted_a[:,:,int(second)]*difference)
        h=np.stack([omega*np.cos(phase),omega*np.sin(phase),delta+shift-frequency],axis=-1)
        r=rotate(r,h,durations[s])
        if s==n-1:
            h=np.zeros_like(r);h[...,2]=delta
            r=rotate(r,h,dark)
            r[...,:2]*=np.exp(-config['gamma']*dark)[...,None]
    return np.clip((1-r[...,2])/2,1e-10,1-1e-10)


def noise_kernels(times,config):
    sigma=config['sigma_intensity'];rho=config['beam_correlation']
    distance=abs(times[:,None]-times[None,:])
    base=sigma*sigma*np.exp(-distance/config['correlation_time'])
    return .5*(1+rho)*base,.5*(1-rho)*base


def finish(p,derivatives,common,difference,kc,kd,config,full=False,counts=None):
    covariance_common=common@kc@common.T
    covariance_difference=difference@kd@difference.T
    if counts is None:counts=config['atoms_per_shot']
    projection=np.diag(p*(1-p)/counts)
    covariance=projection+covariance_common+covariance_difference
    solved=cho_solve(cho_factor(covariance),derivatives)
    information=derivatives.T@solved
    effective=information[0,0]-information[0,1]**2/max(information[1,1],1e-25)
    variance=1/max(effective,1e-25)
    weights=(solved[:,0]-information[0,1]/max(information[1,1],1e-25)*solved[:,1])*variance
    result=dict(variance=float(variance),shot_variance=float(weights@projection@weights),
                common_variance=float(weights@covariance_common@weights),
                difference_variance=float(weights@covariance_difference@weights))
    if full:result.update(weights=weights,probabilities=p,jacobian=derivatives,covariance=covariance,
                          common_jacobian=common,difference_jacobian=difference)
    return result


def evaluate(v,kind,pattern,config,bs,details=False,full=False):
    flags,phases,centers,_=layout(v,pattern,config)
    cost=resources(v,pattern,config)
    peak_cap=config.get('peak_cap',np.sqrt(1+config.get('b_width',.6)**2))
    excess=max(cost['elapsed']/config['time_cap'],cost['exposure']/config['exposure_cap'],cost['peak']/peak_cap)
    if excess>1+1e-10:
        return dict(invalid=True,resources=cost) if details else 1e-4*excess**2
    bs=np.asarray(bs);k=len(flags);nb=len(bs);h=1e-5
    # Parameters: delta,b,c_first,c_second,d_first,d_second.
    nominal=np.zeros((nb,6));nominal[:,1]=bs
    x=np.concatenate([nominal]+[nominal+sign*h*np.eye(6)[j] for j in range(6) for sign in (1,-1)])
    noise=np.zeros((len(x),k,2,6))
    for channel in (0,1):
        for pulse in (0,1):
            noise[:,:,channel,3*pulse:3*pulse+3]=x[:,2+2*channel+pulse,None,None]
    ps=probability(x[:,0],x[:,1],noise,v,kind,flags,phases,config).reshape(13,nb,k)
    jac=np.stack([(ps[1+2*j]-ps[2+2*j])/(2*h) for j in range(6)],axis=-1)
    kc,kd=noise_kernels(centers,config)
    records=[]
    for bi,b in enumerate(bs):
        common=np.zeros((k,2*k));difference=np.zeros((k,2*k))
        for pulse in (0,1):
            common[np.arange(k),2*np.arange(k)+pulse]=jac[bi,:,2+pulse]
            difference[np.arange(k),2*np.arange(k)+pulse]=jac[bi,:,4+pulse]
        row=finish(ps[0,bi],jac[bi,:,:2],common,difference,kc,kd,config,full=full,
                   counts=atom_counts(v,kind,flags,config))
        records.append(dict(b=float(b),**row))
    worst=max(r['variance'] for r in records)
    if details:return dict(variance=worst,records=records,resources=cost)
    return worst


def statistics(v,kind,pattern,b,config,substeps=3,full=False):
    flags,phases,_,_=layout(v,pattern,config)
    centers,_,_=geometry(v,kind,flags,config,substeps)
    k,n=centers.shape;h=1e-5
    noise=np.zeros((1+4*n,k,2,n))
    for channel in (0,1):
        for j in range(n):
            index=1+2*(channel*n+j)
            noise[index,:,channel,j]=h;noise[index+1,:,channel,j]=-h
    ps=probability(0.,b,noise,v,kind,flags,phases,config,substeps)
    sensitivity=((ps[1::2]-ps[2::2])/(2*h)).reshape(2,n,k).transpose(0,2,1)
    matrices=np.zeros((2,k,k*n))
    for channel in (0,1):
        for j in range(k):matrices[channel,j,j*n:(j+1)*n]=sensitivity[channel,j]
    zeros=np.zeros((4,k,2,n))
    shifted=probability([h,-h,0.,0.],[b,b,b+h,b-h],zeros,v,kind,flags,phases,config,substeps)
    derivatives=np.stack([(shifted[0]-shifted[1])/(2*h),(shifted[2]-shifted[3])/(2*h)],axis=-1)
    kc,kd=noise_kernels(centers.ravel(),config)
    result=finish(ps[0],derivatives,*matrices,kc,kd,config,full,atom_counts(v,kind,flags,config))
    result['resources']=resources(v,pattern,config)
    return result
