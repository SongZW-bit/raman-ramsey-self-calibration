"""Validate ratio-control physics, resource caps, and static degeneracy."""
from pathlib import Path
import json
import numpy as np
from drift_study import layout
from two_beam import probability, resources, ratio_matrix, evaluate
from two_beam_validate import lambda_probability

OUT=Path(__file__).resolve().parent/'results'


def main():
    source=json.loads((OUT/'two_beam_quadratic_joint.json').read_text())
    row=next(r for r in source['results'] if r['family']=='shape')
    base=np.array(row['parameters']);pattern=row['pattern'];cfg=source['config']
    config=dict(cfg,ratio_modulation=True,ratio_alternating=True,gamma=0.)
    flags,phases,_,_=layout(base,pattern,config);k=len(flags)
    zero=np.zeros((1,k,2,6));v=np.r_[base,np.zeros(4)]
    old=probability(.0005,.4,zero,base,'shape',flags,phases,dict(cfg,gamma=0.))
    np.testing.assert_allclose(old,probability(.0005,.4,zero,v,'shape',flags,phases,config),atol=1e-13)
    for field in ('exposure','peak','elapsed'):
        np.testing.assert_allclose(resources(v,pattern,config)[field],resources(base,pattern,cfg)[field],atol=1e-12)
    v[-4:]=[.3,-.2,-.25,.1]
    cost=resources(v,pattern,config);peak_cap=np.sqrt(1+cfg['b_width']**2)
    assert cost['peak']>peak_cap
    assert evaluate(v,'shape',pattern,config,[0.],True)['invalid']
    v[4]*=peak_cap/cost['peak']
    cost=resources(v,pattern,config);q=ratio_matrix(v,flags,config)
    exposures=[];peaks=[]
    for b in np.linspace(-.6,.6,1201):
        a=np.sqrt(1+b*b)
        beam1=.5*v[4]*(a-b)*np.exp(-q)
        beam2=.5*v[4]*(a+b)*np.exp(q)
        exposures.append(float(np.sum((beam1+beam2)*v[:2][None,:])))
        peaks.append(float(np.max(beam1+beam2)))
    np.testing.assert_allclose(max(exposures),cost['exposure'],atol=1e-11)
    np.testing.assert_allclose(max(peaks),cost['peak'],atol=1e-11)
    identities=[];h=1e-5;c=.015;d=.02
    for b in (-.6,0.,.6):
        common_noise=zero.copy();common_noise[:,:,0,:]=c;common_noise[:,:,1,:]=d
        speed=np.sqrt((1+c)**2-d*d)
        beff=np.sinh(np.arcsinh(b)+.5*np.log((1+c-d)/(1+c+d)))
        effective=v.copy();effective[4]*=speed
        actual=probability(.0005,b,common_noise,v,'shape',flags,phases,config)
        mapped=probability(.0005,beff,zero,effective,'shape',flags,phases,config)
        plus=zero.copy();minus=zero.copy();plus[:,:,1,:]=h;minus[:,:,1,:]=-h
        dd=(probability(0.,b,plus,v,'shape',flags,phases,config)-probability(0.,b,minus,v,'shape',flags,phases,config))/(2*h)
        db=(probability(0.,b+h,zero,v,'shape',flags,phases,config)-probability(0.,b-h,zero,v,'shape',flags,phases,config))/(2*h)
        error=float(np.max(abs(dd+np.sqrt(1+b*b)*db)))
        assert error<2e-8
        assert np.max(abs(actual-mapped))<1e-12
        identities.append(dict(b=b,derivative_error=error,nonlinear_mapping_error=float(np.max(abs(actual-mapped)))))
    noise=np.random.default_rng(9013).normal(0,.015,(1,k,2,6));physical=[]
    for detuning in (1e3,1e4,1e5):
        errors=[]
        for b in (-.6,0.,.6):
            predicted=probability(.0005,b,noise,v,'shape',flags,phases,config)[0]
            exact=lambda_probability(.0005,b,noise,v,'shape',flags,phases,config,detuning)
            errors.append(float(max(abs(predicted-exact))))
        physical.append(dict(detuning=detuning,error=max(errors)))
    assert physical[-1]['error']<1e-4
    report=dict(zero_modulation_reduction=True,resources=cost,
                resource_dense_check=True,static_identities=identities,lambda_comparison=physical)
    (OUT/'ratio_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
