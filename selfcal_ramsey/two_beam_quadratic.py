"""Quadratic Gaussian-noise surrogate for Raman control design.

Includes Hessian-Hessian covariance and expected binomial variance. Does not
include linear-cubic covariance, hence is not a complete order-sigma^4 model.
The parameter response remains the noise-free local Jacobian.
"""
import numpy as np
from drift_study import layout
from two_beam import probability, noise_kernels, atom_counts, geometry
from two_beam import evaluate as linear_evaluate, statistics as linear_statistics


def corrections(v,kind,pattern,config,bs,step=.001):
    flags,phases,times,_=layout(v,pattern,config);k=len(flags)
    points=[np.zeros(4)];pairs=[]
    for a in range(4):points.extend([step*np.eye(4)[a],-step*np.eye(4)[a]])
    for a in range(4):
        for c in range(a+1,4):
            pairs.append((a,c,len(points)))
            points.extend([step*(sa*np.eye(4)[a]+sc*np.eye(4)[c])
                           for sa,sc in [(1,1),(1,-1),(-1,1),(-1,-1)]])
    points=np.asarray(points);m=len(points);bs=np.atleast_1d(bs)
    noise=np.zeros((m,k,2,6))
    for a in range(4):noise[:,:,a//2,3*(a%2):3*(a%2)+3]=points[:,a,None,None]
    p=probability(0.,np.repeat(bs,m),np.tile(noise,(len(bs),1,1,1)),
                  v,kind,flags,phases,config).reshape(len(bs),m,k)
    hessian=np.zeros((len(bs),k,4,4))
    for a in range(4):hessian[:,:,a,a]=(p[:,1+2*a]+p[:,2+2*a]-2*p[:,0])/step**2
    for a,c,j in pairs:
        hessian[:,:,a,c]=hessian[:,:,c,a]=(p[:,j]-p[:,j+1]-p[:,j+2]+p[:,j+3])/(4*step**2)
    kernels=noise_kernels(times,config);cross=np.zeros((k,k,4,4))
    for channel in range(2):
        cross[:,:,2*channel:2*channel+2,2*channel:2*channel+2]=kernels[channel].reshape(k,2,k,2).transpose(0,2,1,3)
    shift=.5*np.einsum('niab,iab->ni',hessian,cross[np.arange(k),np.arange(k)])
    covariance=.5*np.einsum('niab,ijbc,njcd,ijad->nij',hessian,cross,hessian,cross,optimize=True)
    return shift,covariance


def correct(row,shift,quadratic,counts,kernels):
    p=row['probabilities'];mu=p+shift
    old_projection=p*(1-p)/counts
    noise_cov=row['covariance']-np.diag(old_projection)+quadratic
    projection=(mu*(1-mu)-np.diag(noise_cov))/counts
    if np.any(projection<=0):raise ValueError('Quadratic surrogate has invalid expected binomial variance')
    covariance=noise_cov+np.diag(projection)
    jac=row['jacobian'];solved=np.linalg.solve(covariance,jac)
    information=jac.T@solved
    variance=1/max(information[0,0]-information[0,1]**2/max(information[1,1],1e-25),1e-25)
    weights=(solved[:,0]-information[0,1]/max(information[1,1],1e-25)*solved[:,1])*variance
    common=row['common_jacobian']@kernels[0]@row['common_jacobian'].T
    difference=row['difference_jacobian']@kernels[1]@row['difference_jacobian'].T
    return dict(row,linear_variance=row['variance'],variance=float(variance),weights=weights,
                covariance=covariance,mean_shift=shift,quadratic_covariance=quadratic,
                shot_variance=float(np.sum(weights**2*projection)),
                common_variance=float(weights@common@weights),
                difference_variance=float(weights@difference@weights),
                quadratic_variance=float(weights@quadratic@weights))


def evaluate(v,kind,pattern,config,bs,details=False,full=False):
    result=linear_evaluate(v,kind,pattern,config,bs,True,True)
    if result.get('invalid'):
        return result if details else linear_evaluate(v,kind,pattern,config,bs)
    flags,_,times,_=layout(v,pattern,config)
    counts=atom_counts(v,kind,flags,config);kernels=noise_kernels(times,config)
    shift,covariance=corrections(v,kind,pattern,config,bs)
    rows=[correct(row,shift[i],covariance[i],counts,kernels) for i,row in enumerate(result['records'])]
    result['records']=rows;result['variance']=max(r['variance'] for r in rows)
    return result if details else result['variance']


def statistics(v,kind,pattern,b,config,substeps=3,full=False):
    row=linear_statistics(v,kind,pattern,b,config,substeps,True)
    flags=layout(v,pattern,config)[0]
    times=geometry(v,kind,flags,config,substeps)[0].ravel()
    shift,covariance=corrections(v,kind,pattern,config,[b])
    return correct(row,shift[0],covariance[0],atom_counts(v,kind,flags,config),noise_kernels(times,config))
