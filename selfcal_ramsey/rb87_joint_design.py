"""Detuning and phase design with the same physical intensity-ratio nuisance.

The nuisance is specified at a 1000 MHz reference, then transported through
the optical ratio, not held as a constant normalized light shift. Each arm
has the same allowed detuning set. Local risk is a design diagnostic only.
"""
import argparse
import copy
import json
import time
import numpy as np
from scipy.optimize import minimize
from rb87_multilevel import RamanD1,probability_d1,G1,G2
from rb87_multilevel_pilot import statistics,optical_resources,OUT
from rb87_adaptive_experiment import make_designs
from drift_study import layout


def transport_b(reference,model,b_values):
    result=[]
    for b in b_values:
        e1,e2=reference.fields(b,1.)
        x=e1/e2
        result.append(model.field_product*(model.ap*x+model.am/x))
    return np.array(result)


def diagnose(entry,model,bs):
    d,c=entry["design"],entry["config"];v=d["parameters"]
    flags,phases,_,_=layout(v,d["pattern"],c)
    rows=[]
    for b in bs:
        s=statistics(d,c,b,model)
        rho=probability_d1(0.,b,np.zeros((1,len(flags),2,6)),v,d["family"],flags,phases,c,
                           model=model,return_density=True)
        pop=np.diagonal(rho[0],axis1=-2,axis2=-1).real
        rows.append(dict(b=float(b),variance=s["variance"],shot_variance=s["shot_variance"],
            common_variance=s["common_variance"],difference_variance=s["difference_variance"],
            max_leakage=float(np.max(1-pop[:,G1]-pop[:,G2]))))
    return rows


def optimize_phases(entry,model,bs,maxiter):
    entry=copy.deepcopy(entry)
    v=entry["design"]["parameters"]
    ids=[5,7] if entry["design"]["family"] in ("square","hyper") else [5,6,7,8,9,11]
    start=np.array(v)[ids];best=[np.inf,start.copy()];evaluations=0
    def objective(x):
        nonlocal evaluations
        for j,value in zip(ids,x):v[j]=float(value)
        loss=np.mean([statistics(entry["design"],entry["config"],b,model)["variance"] for b in bs])/1e-8
        evaluations+=1
        if loss<best[0]:best[:]=[loss,np.array(x)]
        if evaluations%25==0:
            print(json.dumps(dict(stage="phase_fit",name=entry["name"],detuning=model.detuning_mhz,
                                  evaluations=evaluations,best_variance=best[0]*1e-8)),flush=True)
        return loss
    initial=objective(start)
    fit=minimize(objective,start,method="L-BFGS-B",bounds=[(x-np.pi,x+np.pi) for x in start],
                 options=dict(maxiter=maxiter,ftol=1e-7,eps=2e-4,maxls=12))
    for j,value in zip(ids,best[1]):v[j]=float(value)
    return entry,dict(initial_variance=initial*1e-8,final_variance=best[0]*1e-8,
                      evaluations=evaluations,success=bool(fit.success),message=str(fit.message))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--detunings",type=float,nargs="+",default=[100.,200.,300.,500.,1000.])
    p.add_argument("--points",type=float,nargs="+",default=[-.6,0.,.6])
    p.add_argument("--optimize",action="store_true")
    p.add_argument("--maxiter",type=int,default=16)
    p.add_argument("--start-from")
    p.add_argument("--include-square",action="store_true")
    p.add_argument("--output",default="rb87_joint_design.json")
    args=p.parse_args();reference=RamanD1();base=make_designs(reference)
    fixed_entries=base[:2]
    if args.include_square:
        square=copy.deepcopy(base[0]);square["name"]="fixed_square"
        square["design"]["family"]="square"
        v=square["design"]["parameters"];v[0]=v[1]=np.pi/(2*v[4]);v[5]=v[6]=v[7]=0.
        fixed_entries=[square]
    previous=json.loads((OUT/args.start_from).read_text()) if args.start_from else None
    resource_keys=["normalized_field_squared_peak","normalized_field_squared_exposure"]
    cap={key:max(optical_resources(e["design"],e["config"],reference)[key] for e in base[:2]) for key in resource_keys}
    result=dict(scope="Exact D1 local-risk design screen, not finite-data adaptive MSE. Physical log-ratio uncertainty and q controls shared across detunings. Nominal Raman coupling held fixed; optical resource caps enforced.",
        reference_detuning_mhz=1000.,reference_b_points=args.points,
        reference_log_field_ratio=[float(np.log(reference.fields(b,1.)[0]/reference.fields(b,1.)[1])) for b in args.points],
        optical_caps=cap,rows=[])
    for detuning in args.detunings:
        model=RamanD1(detuning_mhz=detuning);bs=transport_b(reference,model,args.points)
        envelope=transport_b(reference,model,np.linspace(-.6,.6,121))
        for entry in fixed_entries:
            start=time.perf_counter();entry=copy.deepcopy(entry)
            if previous:
                matches=[r for r in previous["rows"] if r["detuning_mhz"]==detuning and r["name"]==entry["name"]]
                if matches:entry=copy.deepcopy(matches[0]["entry"])
            resource=optical_resources(entry["design"],entry["config"],model,envelope)
            feasible=all(resource[key]<=cap[key]*(1+1e-10) for key in cap)
            if not feasible:continue
            diagnostic=diagnose(entry,model,bs);optimization=None
            if args.optimize:
                entry,optimization=optimize_phases(entry,model,bs,args.maxiter)
            rows=diagnose(entry,model,bs) if args.optimize else diagnostic
            row=dict(detuning_mhz=detuning,name=entry["name"],entry=entry,resources=resource,
                original_rows=diagnostic,rows=rows,mean_variance=float(np.mean([r["variance"] for r in rows])),
                optimization=optimization,elapsed=time.perf_counter()-start)
            result["rows"].append(row);(OUT/args.output).write_text(json.dumps(result,indent=2))
            print(json.dumps({k:row[k] for k in ["detuning_mhz","name","mean_variance","optimization","elapsed"]}),flush=True)
        model.pulse_map.cache_clear()
    best=min(result["rows"],key=lambda r:r["mean_variance"])
    result["best_fixed"]=dict(detuning_mhz=best["detuning_mhz"],name=best["name"],mean_variance=best["mean_variance"])
    (OUT/args.output).write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()
