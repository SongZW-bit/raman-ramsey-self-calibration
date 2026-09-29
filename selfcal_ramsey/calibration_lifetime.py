"""Exact calibration-age benchmarks with an unknown constant mean.

These are Gaussian phase-observation benchmarks, not full Raman simulations
or claims of a novel symmetric interrogation protocol.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'results'


def residual(times, target, tau, variance=1., readout=.1):
    times = np.asarray(times)
    covariance = variance*np.exp(-abs(times[:,None]-times[None,:])/tau)
    covariance += np.eye(len(times))*readout
    cross = variance*np.exp(-abs(times-target)/tau)
    one = np.ones(len(times))
    inv_cross = np.linalg.solve(covariance,cross)
    inv_one = np.linalg.solve(covariance,one)
    weights = inv_cross+inv_one*(1-one@inv_cross)/(one@inv_one)
    error = variance-cross@inv_cross+(1-one@inv_cross)**2/(one@inv_one)
    return float(error),weights


def analytic(r, variance=1., readout=.1):
    bracket = variance*(1.5-2*r+.5*r*r)+readout/2
    causal = (variance*(1.5+.5*r-r-r*r)+readout/2
              -variance**2*r*r*(1-r)**2/(2*(variance*(1-r)+readout)))
    return causal,bracket


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(4517)
    rows=[]
    max_identity_error=0.
    for ratio in (.01,.1,1.,10.):
        for age in (.03,.1,.3,1.,3.):
            tau=1/age
            exact=analytic(np.exp(-age),readout=ratio)
            for j,(name,times,target) in enumerate((('past',[0.,1.],2.),('bracket',[0.,2.],1.))):
                variance,weights=residual(times,target,tau,readout=ratio)
                max_identity_error=max(max_identity_error,abs(variance-exact[j]))
                t=np.array([*times,target])
                cov=np.exp(-abs(t[:,None]-t[None,:])/tau)+ratio*np.eye(3)
                samples=3.7+rng.multivariate_normal(np.zeros(3),cov,size=60000)
                errors=samples[:,2]-samples[:,:2]@weights
                squares=errors**2
                rows.append(dict(protocol=name,age_over_tau=age,readout_over_drift=ratio,
                    weights=weights.tolist(),predicted_mse=variance+ratio,
                    mc_mse=float(squares.mean()),mc_se=float(squares.std(ddof=1)/np.sqrt(len(squares))),
                    bias=float(errors.mean()),unbiasedness_error=float(abs(weights.sum()-1))))
    assert max_identity_error < 1e-12
    max_z=max(abs(z['mc_mse']-z['predicted_mse'])/z['mc_se'] for z in rows)
    assert max_z < 5
    (OUT/'calibration_lifetime.json').write_text(json.dumps(dict(
        model='Linear Gaussian phase measurements, unknown common mean, equal three-shot resources',
        note='Past-only prediction and offline bracketing have different output latency. Neither is a new protocol.',
        formula_max_error=max_identity_error,mc_max_standard_errors=max_z,results=rows),indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
    ages=np.geomspace(.003,5,301)
    for ratio,color in zip((.01,.1,1.,10.),('#007C91','#C05A2B','#6B54A3','#4B7551')):
        r=np.exp(-ages)
        causal,bracket=analytic(r,readout=ratio)
        axes[0].semilogx(ages,100*(1-(bracket+ratio)/(causal+ratio)),color=color,label=f'R / S = {ratio:g}')
    axes[0].set(xlabel='Shot spacing / correlation time',ylabel='Total MSE reduction (%)',
                title='Bracketing versus optimized past calibration',ylim=(0,55))
    axes[0].legend(frameon=False)
    ages2=np.linspace(0,3,301)
    ratio=.1
    axes[1].plot(ages2,1-np.exp(-2*ages2)/(1+ratio),label='Known mean: conditional predictor',color='#007C91')
    axes[1].plot(ages2,2*(1-np.exp(-ages2))+ratio,label='Unknown mean: unbiased subtraction',color='#C05A2B')
    axes[1].set(xlabel='Calibration age / correlation time',ylabel='Residual variance / S',
                title='One calibration, R / S = 0.1')
    axes[1].legend(frameon=False,fontsize=8,loc='lower right')
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(OUT/'calibration_lifetime.png',dpi=180)
    fig.savefig(OUT/'calibration_lifetime.pdf')
    print(json.dumps(dict(formula_max_error=max_identity_error,mc_max_standard_errors=max_z,
                         outputs=['calibration_lifetime.json','calibration_lifetime.png','calibration_lifetime.pdf'])))


if __name__=='__main__':main()
