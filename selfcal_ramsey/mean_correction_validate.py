"""Compare Hessian mean corrections with tensor Gauss-Hermite quadrature."""
from itertools import product
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.hermite import hermgauss
from drift_study import layout
from two_beam import probability
from two_beam_inference import second_order_mean


def main():
    out=Path(__file__).resolve().parent/'results'
    source=json.loads((out/'two_beam_two_allocated.json').read_text())
    design=min((r for r in source['results'] if r['family']=='shape'),key=lambda r:r['variance'])
    v=design['parameters'];cfg=source['config']
    flags,phases,_,_=layout(v,design['pattern'],cfg)
    nodes,weights=hermgauss(7)
    indices=np.array(list(product(range(7),repeat=4)))
    z=np.sqrt(2)*nodes[indices];w=np.prod(weights[indices],axis=1)/np.pi**2
    gap=np.where(flags,v[3],v[2])+(v[0]+v[1])/2
    r=np.exp(-gap/cfg['correlation_time'])
    records=[]
    for sigma in (.015,.03):
        config=dict(cfg,sigma_intensity=sigma)
        noise=np.zeros((len(z),len(flags),2,6))
        for channel in (0,1):
            std=sigma*np.sqrt((1+(-1)**channel*cfg['beam_correlation'])/2)
            noise[:,:,channel,:3]=std*z[:,2*channel,None,None]
            noise[:,:,channel,3:]=std*(r[None,:]*z[:,2*channel,None]+
                  np.sqrt(1-r*r)[None,:]*z[:,2*channel+1,None])[:,:,None]
        for b in (-.6,0.,.6):
            p0=probability(0.,b,np.zeros((1,len(flags),2,6)),v,'shape',flags,phases,config)[0]
            expected=w@probability(0.,b,noise,v,'shape',flags,phases,config)
            correction=second_order_mean(design,config,np.array([b]))[0]
            halved=second_order_mean(design,config,np.array([b]),.05)[0]
            error=float(max(abs(expected-p0-correction)))
            assert max(abs(correction-halved))<1e-6
            assert error<3e-5
            records.append(dict(sigma=sigma,b=b,max_mean_shift=float(max(abs(expected-p0))),
                                residual_after_second_order=error,
                                finite_difference_step_error=float(max(abs(correction-halved)))))
    (out/'mean_correction_validation.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
