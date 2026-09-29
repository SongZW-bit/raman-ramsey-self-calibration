"""Audit batched nonlinear fitting against independent scipy minimization."""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.optimize import minimize
from two_beam_monte_carlo import lookup,generate
from two_beam_inference import NonlinearEstimator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corrected',action='store_true')
    parser.add_argument('--width',type=float,default=.8)
    parser.add_argument('--source',default='two_beam_two_allocated.json')
    parser.add_argument('--family',default='shape')
    parser.add_argument('--output')
    args=parser.parse_args()
    out=Path(__file__).resolve().parent/'results'
    source=json.loads((out/args.source).read_text())
    design=min((r for r in source['results'] if r['family']==args.family),key=lambda r:r['variance'])
    config=dict(design.get('config',source['config']),integer_atoms=True)
    grid=np.linspace(-args.width,args.width,401)
    estimator=NonlinearEstimator(design,config,grid,lookup(design,config,grid),args.corrected)
    truth=np.array([[.5,-.55],[.5,0.],[.5,.55]])
    exact=estimator.mean(truth)
    # Parameter-dependent covariance can move Gaussian ML slightly even for
    # data exactly equal to the mean; check mean inversion separately.
    recovered=[]
    for row,y in zip(truth,exact):
        fit=minimize(lambda x:np.sum((estimator.mean(x[None,:])[0]-y)**2),
                     row+[.1,.01],method='BFGS',options=dict(gtol=1e-12))
        recovered.append(float(np.max(abs(estimator.mean(fit.x[None,:])[0]-y))))
    assert max(recovered)<1e-6
    y=generate(design,config,-.6,200,np.random.default_rng(6998),.0005)
    fitted,loss=estimator.fit(y)
    chosen=np.unique(np.r_[np.argsort(abs(fitted[:,0]-.0005))[-8:],np.arange(4)])
    discrepancies=[]
    for index in chosen:
        seed=fitted[index]*[1000.,1.]
        fit=minimize(lambda x:estimator.loss(x[None,:],y[index:index+1])[0],seed,
                     method='L-BFGS-B',bounds=[(-10.,10.),(-args.width,args.width)],
                     options=dict(maxiter=300,ftol=1e-13,gtol=1e-7))
        discrepancies.append(float(loss[index]-fit.fun))
    result=dict(noiseless_population_errors=recovered,
                independent_scipy_loss_improvements=discrepancies,
                max_loss_improvement=max(discrepancies),
                note='Checks local refinement, not global maximum likelihood or covariance accuracy.')
    result.update(second_order_mean=args.corrected,b_fit_width=args.width,source=args.source,family=args.family)
    filename=args.output or ('inference_corrected_validation.json' if args.corrected else 'inference_validation.json')
    (out/filename).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    assert max(discrepancies)<1e-3


if __name__=='__main__':main()
