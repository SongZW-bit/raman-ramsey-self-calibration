"""Fixed diagnostic phase coding: local precision versus remote ambiguity.

The diagnostic uses truth-covariance Mahalanobis mean distances, not a
global hypothesis-testing bound or an adaptive policy.
"""
from pathlib import Path
import json
import numpy as np
from drift_study import layout
from two_beam import evaluate,statistics,probability

OUT=Path(__file__).resolve().parent/'results'


def main():
    source=json.loads((OUT/'two_beam_two_allocated.json').read_text())
    original=min((r for r in source['results'] if r['family']=='shape'),key=lambda r:r['variance'])
    v=original['parameters'];cfg=dict(source['config'],integer_atoms=True)
    flags,phases,_,_=layout(v,original['pattern'],cfg)
    assert not np.any(flags[[0,4,8,12]])
    bb,dd=np.meshgrid(np.linspace(-.8,-.65,51),np.linspace(.002,.008,81))
    bs=np.r_[-.6,bb.ravel()];ds=np.r_[.0005,dd.ravel()]
    noise=np.zeros((len(bs),len(flags),2,6))
    rows=[]
    for angle in np.linspace(-np.pi,np.pi,41):
        coded=phases.copy();coded[[0,4,8,12]]+=angle
        pattern=dict(long_flags=flags.tolist(),phases=coded.tolist())
        local=evaluate(v,'shape',pattern,cfg,np.linspace(-.6,.6,41))
        stats=statistics(v,'shape',pattern,-.6,cfg,3,True)
        p=probability(ds,bs,noise,v,'shape',flags,coded,cfg)
        residual=p[1:]-p[0]
        distance=np.einsum('ni,ij,nj->n',residual,np.linalg.inv(stats['covariance']),residual)
        index=np.argmin(distance)
        rows.append(dict(angle=float(angle),local_variance=float(local),
                         minimum_remote_distance_squared=float(distance[index]),
                         remote_b=float(bs[index+1]),remote_delta=float(ds[index+1])))
    # Predefined selection: strongest separation with <=10% local variance cost.
    zero=min(rows,key=lambda r:abs(r['angle']))
    best=max((r for r in rows if r['local_variance']<=1.1*zero['local_variance']),
             key=lambda r:r['minimum_remote_distance_squared'])
    coded=phases.copy();coded[[0,4,8,12]]+=best['angle']
    pattern=dict(long_flags=flags.tolist(),phases=coded.tolist())
    dense=[statistics(v,'shape',pattern,b,cfg,3) for b in np.linspace(-.6,.6,81)]
    design=dict(original,pattern=pattern,readout='two-sentinel',variance=max(s['variance'] for s in dense),
                diagnostic=best,source_design='two_beam_two_allocated.json')
    result=dict(config=cfg,results=[design],scan=rows,
                scope='Fixed controls; diagnostic region is b in [-.8,-.65], delta in [.002,.008]. Not a global ambiguity guarantee.')
    (OUT/'two_beam_discrimination.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(original=zero,selected=best,finite_pulse_variance=design['variance']),indent=2))


if __name__=='__main__':main()
