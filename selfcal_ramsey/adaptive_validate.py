"""Independent control-insertion and numerical inference checks for the pilot."""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.optimize import minimize
from drift_study import layout
from two_beam import probability,resources
from two_beam_adaptive import branch_config
from two_beam_monte_carlo import lookup,generate
from two_beam_inference import NonlinearEstimator
from two_beam_optimize import reshape

OUT=Path(__file__).resolve().parent/'results'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_adaptive_pilot.json')
    parser.add_argument('--output',default='adaptive_validation.json')
    args=parser.parse_args()
    data=json.loads((OUT/args.source).read_text())
    design=data['design'];config=data['config'];v=np.array(design['parameters'])
    flags,phases,_,_=layout(v,design['pattern'],config)
    rng=np.random.default_rng(42670)
    noise=rng.normal(0,.02,(3,len(flags),2,6))
    offsets=np.linspace(-.4,.4,len(flags))
    changed=dict(config,frequency_offsets=offsets.tolist())
    p=probability(.001,.2,noise,v,design['family'],flags,phases,changed)
    index=10 if design['family']=='shape' else 6
    errors=[]
    for i,offset in enumerate(offsets):
        vv=v.copy();vv[index]+=offset
        independent=probability(.001,.2,noise,vv,design['family'],flags,phases,config)
        errors.append(float(np.max(abs(p[:,i]-independent[:,i]))))
    assert max(errors)<1e-12
    assert resources(v,design['pattern'],config)==resources(v,design['pattern'],changed)
    segment_phases=np.zeros((len(flags),6));segment_phases[:,1:5]=rng.normal(0,.3,(len(flags),4))
    changed_phase=dict(config,segment_phase_offsets=segment_phases.tolist())
    actual=probability(.001,.2,noise,v,design['family'],flags,phases,changed_phase)
    size=8 if design['family']=='hyper' else 12
    shaped=np.array(reshape(v[:size],design['family'],'shape')+v[size:].tolist())
    phase_errors=[]
    for i in range(len(flags)):
        vv=shaped.copy();vv[6:10]+=segment_phases[i,1:5]
        independent=probability(.001,.2,noise,vv,'shape',flags,phases,config)
        phase_errors.append(float(np.max(abs(actual[:,i]-independent[:,i]))))
    assert max(phase_errors)<1e-12
    assert resources(v,design['pattern'],config)==resources(v,design['pattern'],changed_phase)
    cfg=branch_config(config,data['offsets'][-1],data['calibration_shots'])
    grid=np.linspace(-.8,.8,161)
    estimator=NonlinearEstimator(design,cfg,grid,lookup(design,cfg,grid),True)
    y=generate(design,cfg,.6,40,rng,.0005)
    fits,loss=estimator.fit(y)
    improvements=[]
    for i in np.unique(np.r_[np.argsort(abs(fits[:,0]-.0005))[-4:],np.arange(4)]):
        fit=minimize(lambda x:estimator.loss(x[None,:],y[i:i+1])[0],fits[i]*[1000.,1.],
                     method='L-BFGS-B',bounds=[(-10,10),(-.8,.8)],
                     options=dict(maxiter=300,ftol=1e-13,gtol=1e-7))
        improvements.append(float(loss[i]-fit.fun))
    assert max(improvements)<1e-3
    result=dict(source=args.source,per_shot_frequency_insertion_max_error=max(errors),
                phase_control_embedding_max_error=max(phase_errors),resource_invariance=True,
                independent_scipy_loss_improvements=improvements,
                scope='Control insertion and local inference refinement, not an exact adaptive-risk or global-likelihood certificate.')
    (OUT/args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
