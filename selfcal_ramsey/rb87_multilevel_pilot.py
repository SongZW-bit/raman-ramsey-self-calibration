"""Local-information screening using exact 16-state D1 master equations.

This deliberately reports an oracle separately from an adaptive experiment.
It cannot establish a finite-data adaptive advantage.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from rb87_multilevel import RamanD1, probability_d1, G1, G2
from two_beam import geometry, atom_counts, finish, noise_kernels, control_size
from drift_study import layout
from two_beam_adaptive import branch_config

OUT = Path(__file__).resolve().parent/"results"


def physical_design(design, config, model):
    design = dict(design)
    v = np.array(design["parameters"])
    ratio_scale = (model.ap-model.am)*model.field_product
    if config.get("ratio_modulation",False):
        v[-4:] /= ratio_scale
    design["parameters"] = v.tolist()
    return design,dict(config)


def statistics(design,config,b,model,step=2e-4,substeps=1,probability_fn=probability_d1):
    v = np.asarray(design["parameters"])
    flags,phases,_,_ = layout(v,design["pattern"],config)
    centers,_,_ = geometry(v,design["family"],flags,config,substeps)
    n=6*substeps; k=len(flags)
    noise=np.zeros((1+4*n,k,2,n))
    for channel in (0,1):
        for s in range(n):
            ix=1+2*(channel*n+s)
            noise[ix,:,channel,s]=step
            noise[ix+1,:,channel,s]=-step
    ps=probability_fn(0.,b,noise,v,design["family"],flags,phases,config,substeps,model)
    sensitivity=((ps[1::2]-ps[2::2])/(2*step)).reshape(2,n,k).transpose(0,2,1)
    matrices=np.zeros((2,k,k*n))
    for channel in (0,1):
        for j in range(k):
            matrices[channel,j,j*n:(j+1)*n]=sensitivity[channel,j]
    shifted=probability_fn([step,-step,0.,0.],[b,b,b+step,b-step],np.zeros((4,k,2,n)),
        v,design["family"],flags,phases,config,substeps,model)
    derivatives=np.stack([(shifted[0]-shifted[1])/(2*step),(shifted[2]-shifted[3])/(2*step)],axis=-1)
    kc,kd=noise_kernels(centers.ravel(),config)
    result=finish(ps[0],derivatives,*matrices,kc,kd,config,True,atom_counts(v,design["family"],flags,config))
    return {key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in result.items()}


def optical_resources(design,config,model,b_values=None):
    from two_beam import ratio_matrix
    v=np.asarray(design["parameters"])
    flags,_,_,resources=layout(v,design["pattern"],config)
    ratios=ratio_matrix(v,flags,config)
    rows=[]
    if b_values is None:
        b_values=np.linspace(-.6,.6,121)
    for b in b_values:
        e=np.array([[sum(x*x for x in model.fields(b,v[4],q)) for q in pair] for pair in ratios])
        rows.append([np.sum(e*np.array(v[:2])),e.max()])
    worst=np.max(rows,axis=0)
    return dict(atoms=int(atom_counts(v,design["family"],flags,config).sum()),
        elapsed=resources["elapsed"],normalized_field_squared_exposure=float(worst[0]),
        normalized_field_squared_peak=float(worst[1]),
        scope="Sampled envelope over supplied unknown-b domain; common dipole and geometry factor omitted")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--points",type=float,nargs="+",default=[-.6,.6])
    p.add_argument("--detuning",type=float,default=1000.)
    p.add_argument("--branches",type=int,nargs="+")
    p.add_argument("--output",default="rb87_multilevel_pilot.json")
    args=p.parse_args()
    model=RamanD1(detuning_mhz=args.detuning)
    fixed=json.loads((OUT/"two_beam_fixed_band_mc.json").read_text())
    adaptive=json.loads((OUT/"two_beam_adaptive_phase_band.json").read_text())
    designs=[]
    for entry in fixed["designs"]:
        design,config=physical_design(entry["design"],entry["config"],model)
        designs.append(("fixed_"+design["family"],design,config))
    ad,ac=physical_design(adaptive["design"],adaptive["config"],model)
    for i,control in enumerate(adaptive["offsets"]):
        if args.branches is None or i in args.branches:
            designs.append(("branch_"+str(i),ad,branch_config(ac,control,adaptive["calibration_shots"])))
    result=dict(model=model.metadata(),scope="Frozen controls, physical ratio scaling; local Gaussian information and oracle branch diagnostic only. Not an adaptive finite-data MSE comparison.",
                designs=[dict(name=name,design=d,config=c,resources=optical_resources(d,c,model)) for name,d,c in designs],rows=[])
    for b in args.points:
        for name,design,config in designs:
            start=time.perf_counter()
            stat=statistics(design,config,b,model)
            row=dict(b=b,name=name,statistics=stat,elapsed=time.perf_counter()-start)
            result["rows"].append(row)
            (OUT/args.output).write_text(json.dumps(result,indent=2))
            print(json.dumps(dict(b=b,name=name,variance=stat["variance"],elapsed=row["elapsed"])),flush=True)


if __name__=="__main__":
    main()
