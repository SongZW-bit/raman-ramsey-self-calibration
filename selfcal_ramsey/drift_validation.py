"""Independent checks of the correlated-noise estimator and its gradient."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from drift_study import layout
from drift_continuous import probability, pulse_samples, statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    data=json.loads((OUT/'drift_schedules.json').read_text())
    config=data['config']
    chosen=min((z for z in data['results'] if z['family']=='square'),key=lambda z:z['variance'])
    v=chosen['parameters'];pattern=chosen['pattern']
    series=[]
    for sub in (1,2,4,8,16):
        stats=statistics(v,'square',pattern,.6,config,sub)
        series.append(dict(segments_per_pulse=3*sub,variance=stats['variance']))
    s=statistics(v,'square',pattern,.3,config,4)
    a=s['jacobian'];w=s['weights'];cov=s['covariance']
    direct=float(w@cov@w)
    assert np.max(abs(w@a-np.array([1.,0.])))<1e-8
    assert abs(direct-s['variance'])<1e-15
    # A tangent perturbation of both signal and covariance checks the analytic
    # envelope-theorem gradient, independently of the physical optimizer.
    rng=np.random.default_rng(12342)
    da=rng.normal(size=a.shape)*.01
    m=rng.normal(size=cov.shape)
    dc=(m+m.T)*1e-8
    def variance(t):
        at=a+t*da;ct=cov+t*dc
        return np.linalg.inv(at.T@np.linalg.solve(ct,at))[0,0]
    lam=np.linalg.solve(a.T@np.linalg.solve(cov,a),[1.,0.])
    analytic=w@dc@w-2*lam@da.T@w
    h=1e-3
    numerical=(variance(h)-variance(-h))/(2*h)
    assert abs(analytic-numerical)<1e-12
    # Finite-pulse intensity dependence cross-checked against matrix exponentials.
    flags,phases,_,_=layout(v,pattern,config)
    centers,angles=pulse_samples(v,'square',flags,config,2)
    noise=rng.normal(0,.03,centers.shape)
    delta=.0007;b=.4
    p=probability(delta,b,noise,v,'square',flags,phases,config,2)[0]
    pauli=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
    errs=[];n=len(angles)//2
    for j in range(len(flags)):
        rho=np.diag([1.,0.]).astype(complex)
        dark=v[3] if flags[j] else v[2]
        for step,angle in enumerate(angles):
            second=step>=n
            phi=angle+(phases[j]+v[5]+v[7]*(not flags[j]) if second else 0.)
            intensity=v[4]*(1+noise[j,step])
            hamiltonian=np.einsum('i,ijk->jk',[intensity*np.cos(phi),intensity*np.sin(phi),delta+b*intensity-v[6]],pauli)/2
            u=expm(-1j*hamiltonian*v[int(second)]/n)
            rho=u@rho@u.conj().T
            if step==n-1:
                u=expm(-.5j*delta*pauli[2]*dark)
                rho=u@rho@u.conj().T
                rho[0,1]*=np.exp(-config['gamma']*dark)
                rho[1,0]*=np.exp(-config['gamma']*dark)
        errs.append(abs(p[j]-rho[1,1].real))
    assert max(errs)<1e-12
    result=dict(pulse_resolution=series,unbiasedness_residual=(w@a-[1.,0.]).tolist(),
                covariance_identity_error=abs(direct-s['variance']),
                gradient_analytic=float(analytic),gradient_numeric=float(numerical),
                matrix_probability_error=float(max(errs)),
                asymptotic_scope='Local linear nuisance projection, not a global quantum bound.')
    (OUT/'drift_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
