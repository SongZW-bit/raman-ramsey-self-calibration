"""Optimize shot ordering at identical pulse controls and resource use."""
from pathlib import Path
import json
import numpy as np
from drift_study import evaluate, layout
from drift_continuous import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    src=json.loads((OUT/'drift_schedules.json').read_text())
    config=src['config'];bs=np.linspace(-.6,.6,13)
    results=[]
    for kind in ('square',):
        for nshort in (8,10,12):
            original=next(z for z in src['results'] if z['family']==kind and z['pattern']=='distributed'+str(nshort))
            v=original['parameters']
            flags,phases,_,cost=layout(v,original['pattern'],config)
            best_order=np.arange(len(flags))
            rng=np.random.default_rng(267+nshort)
            def make(order):return dict(long_flags=flags[order].tolist(),phases=phases[order].tolist(),name=f'optimized_{nshort}')
            best=evaluate(v,kind,make(best_order),config,bs)
            for restart in range(5):
                order=np.arange(len(flags)) if restart==0 else (rng.permutation(len(flags)) if restart<3 else best_order.copy())
                value=evaluate(v,kind,make(order),config,bs)
                for iteration in range(700):
                    proposal=order.copy();i,j=rng.choice(len(flags),2,replace=False)
                    proposal[i],proposal[j]=proposal[j],proposal[i]
                    val=evaluate(v,kind,make(proposal),config,bs)
                    temperature=.08*value*(1-iteration/700)**2
                    if val<value or rng.random()<np.exp(min(0.,(value-val)/max(temperature,1e-30))):order,value=proposal,val
                    if value<best:best_order,best=order.copy(),value
            pattern=make(best_order)
            checks=[statistics(v,kind,pattern,b,config,8) for b in np.linspace(-.6,.6,81)]
            worst=max(checks,key=lambda z:z['variance'])
            row=dict(family=kind,pattern=pattern,parameters=v,variance=worst['variance'],
                     resources=cost,prior_variance=original['variance'],dense_variances=[z['variance'] for z in checks])
            results.append(row)
            (OUT/'drift_ordered.json').write_text(json.dumps(dict(config=config,results=results),indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in row.items() if k!='dense_variances'}),flush=True)


if __name__=='__main__':main()
