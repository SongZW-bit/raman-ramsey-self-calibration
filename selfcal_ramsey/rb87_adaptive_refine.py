"""Reoptimize phase controls of fixed and posterior-selectable D1 branches.

All branches share their observed calibration prefix. Phase optimization
changes no optical powers, atom counts, timing, or nominal frequency steps.
"""
import argparse
import copy
import json
import time
import numpy as np
from scipy.optimize import minimize
from rb87_multilevel import RamanD1
from rb87_multilevel_pilot import statistics,OUT
from rb87_adaptive_experiment import make_designs


def main():
    p=argparse.ArgumentParser();p.add_argument("--maxiter",type=int,default=25)
    p.add_argument("--output",default="rb87_adaptive_refined.json")
    args=p.parse_args();model=RamanD1();base=make_designs(model)
    result=dict(model=model.metadata(),scope="Full D1 phase reoptimization, frozen optical ratio and timing. Fixed-bank reference optimized over prior; adaptive branches optimized over local nuisance bands. Local variance only.",entries=[])
    domains=[("branch_0",np.linspace(-.6,.6,7)),("branch_1",[-.68,-.6,-.52]),("branch_2",[-.5,-.4,-.3]),
             ("branch_3",[-.3,-.2,-.1]),("branch_4",[-.1,0.,.1]),("branch_5",[.1,.2,.3]),
             ("branch_6",[.3,.4,.5]),("branch_7",[.52,.6,.68])]
    for name,bs in domains:
        entry=copy.deepcopy(next(e for e in base if e["name"]==name));d,c=entry["design"],entry["config"]
        offsets=np.array(c["segment_phase_offsets"]);x0=offsets[2,1:5].copy()
        best=[np.inf,x0.copy()];neval=0;start=time.perf_counter()
        def objective(x):
            nonlocal neval
            offsets[2:,1:5]=x;c["segment_phase_offsets"]=offsets.tolist()
            risk=np.mean([statistics(d,c,b,model)["variance"] for b in bs])/1e-8
            neval+=1
            if risk<best[0]:best[:]=[risk,np.array(x)]
            if neval%40==0:print(json.dumps(dict(name=name,evaluations=neval,best_variance=best[0]*1e-8)),flush=True)
            return risk
        initial=objective(x0)
        fit=minimize(objective,x0,method="L-BFGS-B",bounds=[(x-np.pi,x+np.pi) for x in x0],
                     options=dict(maxiter=args.maxiter,ftol=1e-7,eps=2e-4,maxls=10))
        offsets[2:,1:5]=best[1];c["segment_phase_offsets"]=offsets.tolist()
        row=dict(name=name,entry=entry,domain=list(bs),initial_variance=initial*1e-8,variance=best[0]*1e-8,
                 success=bool(fit.success),message=str(fit.message),evaluations=neval,elapsed=time.perf_counter()-start)
        result["entries"].append(row);(OUT/args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!="entry"}),flush=True)


if __name__=="__main__":main()
