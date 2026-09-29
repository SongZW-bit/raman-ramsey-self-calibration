"""Multistart inference with exact noise-free means and Gaussian drift covariance.

The covariance retains the small-noise, delta=0 approximation. This is not
an exact marginal likelihood for nonlinear OU-driven binomial observations.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from drift_study import layout
from drift_monte_carlo import fit_samples
from two_beam import probability


def second_order_mean(design,config,grid,step=.1):
    """Trace(Hessian K)/2 using four pulse-center covariance directions.

    Noise is constant inside each pulse here; trajectory simulations retain
    finite-pulse fluctuations. The step multiplies the physical noise SD.
    """
    v=design['parameters'];kind=design['family']
    flags,phases,_,_=layout(v,design['pattern'],config)
    k=len(flags);nb=len(grid)
    gap=(v[0]+v[1])/2+np.where(flags,v[3],v[2])
    corr=np.exp(-gap/config['correlation_time'])
    factors=np.zeros((k,4,4))
    for channel in range(2):
        std=config['sigma_intensity']*np.sqrt((1+(-1)**channel*config['beam_correlation'])/2)
        j=2*channel
        factors[:,j,j]=std;factors[:,j+1,j]=std*corr
        factors[:,j+1,j+1]=std*np.sqrt(1-corr*corr)
    noise=np.zeros((9,k,2,6))
    for direction in range(4):
        for channel in range(2):
            for pulse in range(2):
                z=step*factors[:,2*channel+pulse,direction,None]
                noise[1+2*direction,:,channel,3*pulse:3*pulse+3]=z
                noise[2+2*direction,:,channel,3*pulse:3*pulse+3]=-z
    ps=probability(0.,np.repeat(grid,9),np.tile(noise,(nb,1,1,1)),v,kind,flags,phases,config).reshape(nb,9,k)
    return .5*np.sum(ps[:,1::2]+ps[:,2::2]-2*ps[:,0,None,:],axis=1)/(step*step)


class NonlinearEstimator:
    def __init__(self,design,config,grid,lookup,mean_correction=False):
        self.design=design;self.config=config;self.grid=grid;self.lookup=lookup
        self.precision=CubicSpline(grid,lookup[2])
        self.logdet=CubicSpline(grid,lookup[3])
        self.p_spline=CubicSpline(grid,lookup[0])
        self.a_spline=CubicSpline(grid,lookup[1])
        self.flags,self.phases,_,_=layout(design['parameters'],design['pattern'],config)
        self.correction=CubicSpline(grid,second_order_mean(design,config,grid)) if mean_correction else None
        if mean_correction:
            corrected_p=lookup[0]+self.correction(grid)
            self.lookup=(corrected_p,*lookup[1:])
            self.p_spline=CubicSpline(grid,corrected_p)

    def mean(self,x):
        mean=probability(x[:,0]*.001,x[:,1],np.zeros((len(x),len(self.flags),2,6)),
            self.design['parameters'],self.design['family'],self.flags,self.phases,self.config)
        return mean if self.correction is None else mean+self.correction(x[:,1])

    def loss(self,x,y):
        r=y-self.mean(x);p=self.precision(x[:,1])
        return np.einsum('ni,nij,nj->n',r,p,r)+self.logdet(x[:,1])

    def refine(self,x,y,iterations=70):
        x=x.copy();loss=self.loss(x,y)
        all_x=x.copy();all_loss=loss.copy();active=np.arange(len(x))
        for _ in range(iterations):
            mu=self.mean(x);r=y-mu;p=self.precision(x[:,1])
            h=np.array([.002,2e-5])
            jac=np.stack([(self.mean(x+h[j]*np.eye(2)[j])-self.mean(x-h[j]*np.eye(2)[j]))/(2*h[j])
                          for j in range(2)],axis=-1)
            pr=np.einsum('nij,nj->ni',p,r)
            grad=-2*np.einsum('nik,ni->nk',jac,pr)
            grad[:,1]+=np.einsum('ni,nij,nj->n',r,self.precision(x[:,1],1),r)+self.logdet(x[:,1],1)
            hes=2*np.einsum('nik,nij,njl->nkl',jac,p,jac)
            hes+=np.eye(2)[None,:,:]*1e-8
            step=-np.linalg.solve(hes,grad[...,None])[...,0]
            # At a nuisance boundary, optimize the free frequency coordinate;
            # clipping an unconstrained coupled step is not a constrained solve.
            blocked=((x[:,1]<=self.grid[0]+1e-10)&(step[:,1]<0))|((x[:,1]>=self.grid[-1]-1e-10)&(step[:,1]>0))
            step[blocked,1]=0.
            step[blocked,0]=-grad[blocked,0]/hes[blocked,0,0]
            step/=np.maximum(1,np.max(abs(step)/[2.,.15],axis=1))[:,None]
            candidate=x.copy();candidate_loss=loss.copy();alpha=np.ones(len(x))
            accepted=np.zeros(len(x),bool)
            for _ in range(10):
                ids=np.flatnonzero(~accepted)
                if not len(ids):break
                trial=np.clip(x[ids]+alpha[ids,None]*step[ids],[-10.,self.grid[0]],[10.,self.grid[-1]])
                val=self.loss(trial,y[ids]);ok=val<loss[ids]
                candidate[ids[ok]]=trial[ok];candidate_loss[ids[ok]]=val[ok]
                accepted[ids[ok]]=True;alpha[ids[~ok]]*=.5
            improvement=loss-candidate_loss
            all_x[active]=candidate;all_loss[active]=candidate_loss
            keep=improvement>=1e-7
            if not np.any(keep):break
            x,loss,y=candidate[keep],candidate_loss[keep],y[keep]
            active=active[keep]
        return all_x,all_loss

    def fit(self,y,starts=7,batch_size=300):
        linear=fit_samples(y,self.grid,self.lookup)
        fits=[];losses=[]
        for begin in range(0,len(y),batch_size):
            data=y[begin:begin+batch_size];n=len(data)
            # Include the profiled linear solution and a global nuisance grid.
            bg=np.linspace(self.grid[0],self.grid[-1],starts-1)
            r=data[:,None,:]-self.p_spline(bg)[None,:,:]
            a=self.a_spline(bg);p=self.precision(bg)
            pa=np.einsum('gij,gj->gi',p,a)
            dg=np.einsum('ngi,gi->ng',r,pa)/np.einsum('gi,gi->g',a,pa)
            seed=np.empty((n,starts,2));seed[:,0]=linear[begin:begin+n]*[1000.,1.]
            seed[:,1:,0]=np.clip(dg*1000.,-10.,10.);seed[:,1:,1]=bg
            x,loss=self.refine(seed.reshape(-1,2),np.repeat(data,starts,axis=0))
            x=x.reshape(n,starts,2);loss=loss.reshape(n,starts)
            which=np.argmin(loss,axis=1)
            fits.append(x[np.arange(n),which]*[.001,1.]);losses.append(loss[np.arange(n),which])
        return np.concatenate(fits),np.concatenate(losses)
