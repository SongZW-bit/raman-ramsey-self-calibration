"""Independent covariance and stochastic-innovation checks."""
from pathlib import Path
import json
import numpy as np


def main():
    rng=np.random.default_rng(5821)
    rows=[]
    times=np.cumsum(rng.uniform(.1,2.,32))
    for tau in (.2,2.,20.,200.):
        s=.09
        v=np.exp(-times/tau)
        k=s*np.exp(-abs(times[:,None]-times[None,:])/tau)
        q=k-s*np.outer(v,v)
        dt=np.diff(np.r_[0.,times])
        innovations=np.sqrt(-s*np.expm1(-2*dt/tau))
        lower=np.tril(np.exp(-np.maximum(times[:,None]-times[None,:],0)/tau)*innovations[None,:])
        error=float(np.max(abs(q-lower@lower.T)))
        assert error<1e-14
        assert np.linalg.eigvalsh(q)[0]>0
        h=rng.normal(size=len(times));h-=h.mean()
        variance=float(h@q@h)
        assert variance>0 and abs(h.sum())<1e-13
        samples=rng.normal(size=(100000,len(times)))@lower.T@h
        relative=abs(samples.var(ddof=1)/variance-1)
        assert relative<.025
        rows.append(dict(tau=tau,factorization_error=error,
                         zero_static_kernel_innovation_variance=variance,
                         monte_carlo_relative_error=float(relative)))
    h=rng.normal(size=len(times));h-=h.mean()
    cumulative=np.cumsum(h)[:-1]
    coefficient=2*.09*np.sum(cumulative**2*np.diff(times))
    slow=[]
    for tau in (1e3,1e4,1e5):
        exact=float(h@(.09*np.exp(-abs(times[:,None]-times[None,:])/tau))@h)
        slow.append(dict(tau=tau,exact=exact,asymptotic=coefficient/tau,
                         relative_error=abs(exact/(coefficient/tau)-1)))
    assert slow[-1]['relative_error']<.001
    out=Path(__file__).resolve().parent/'results'/'innovation_validation.json'
    result=dict(conditional_covariance_checks=rows,slow_drift_checks=slow)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
