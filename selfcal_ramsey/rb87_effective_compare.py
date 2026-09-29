"""Matched effective two-level mechanism versus the complete D1 manifold.

The qubit retains the actual multilevel polarizabilities and Raman matrix
element. It omits scattering and leakage, so it explains the coherent
mechanism but is not expected to predict full-D1 precision identically.
"""
import argparse
import json
import numpy as np
from model import rotate
from rb87_multilevel import RamanD1,probability_d1
from rb87_multilevel_pilot import statistics,OUT
from rb87_joint_design import transport_b
from drift_study import layout
from two_beam import geometry,ratio_matrix


def probability_effective(delta,b,noise,v,kind,flags,phases,config,substeps=1,model=None):
    model=RamanD1() if model is None else model
    noise=np.asarray(noise);ns,k,_,total=noise.shape
    delta=np.broadcast_to(np.asarray(delta).reshape(-1,1),(ns,k))
    bs=np.broadcast_to(np.asarray(b).reshape(-1,1),(ns,k))
    # fields() defines the same physical nuisance as the full D1 model.
    base=np.array([model.fields(float(bb),1.) for bb in bs[:,0]])**2
    ratios=ratio_matrix(v,flags,config)
    e1=base[:,None,0,None]*np.exp(ratios)[None,:,:]
    e2=base[:,None,1,None]*np.exp(-ratios)[None,:,:]
    shift0=model.ap*e1+model.am*e2
    susceptibility=model.ap*e1-model.am*e2
    extra=10 if kind in ("shape","flex") else 6
    frequency=v[extra]+np.asarray(config.get("frequency_offsets",0.))
    offsets=np.asarray(config.get("segment_phase_offsets",np.zeros((k,6))))
    _,angles,durations=geometry(v,kind,flags,config,substeps)
    r=np.zeros((ns,k,3));r[...,2]=1.
    dark=np.where(flags,v[3],v[2])[None,:]
    for s,angle in enumerate(angles):
        second=int(s>=total//2)
        phase=angle+offsets[None,:,s//substeps]+second*(phases[None,:]+v[5]+v[extra+1]*(~flags)[None,:])
        common,difference=noise[:,:,0,s],noise[:,:,1,s]
        omega=v[4]*np.sqrt((1+common)**2-difference**2)
        shift=v[4]*(shift0[:,:,second]*(1+common)+susceptibility[:,:,second]*difference)
        h=np.stack([omega*np.cos(phase),omega*np.sin(phase),delta+shift-frequency],axis=-1)
        r=rotate(r,h,durations[s])
        if s==total//2-1:
            h=np.zeros_like(r);h[...,2]=delta;r=rotate(r,h,dark)
            r[...,:2]*=np.exp(-model.gamma_dark*dark)[...,None]
    return np.clip((1-r[...,2])/2,1e-10,1-1e-10)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="rb87_joint_phase_refined.json")
    p.add_argument("--output",default="rb87_effective_comparison.json")
    args=p.parse_args();source=json.loads((OUT/args.source).read_text())
    reference=RamanD1();result=dict(scope="Identical frozen controls and physical nuisance in both models. Effective qubit retains D1 polarizabilities but omits scattering and leakage. Local variance, not Monte Carlo MSE.",rows=[],coherent_checks=[])
    for row in source["rows"]:
        model=RamanD1(detuning_mhz=row["detuning_mhz"])
        d,c=row["entry"]["design"],row["entry"]["config"]
        brefs=np.linspace(-.6,.6,7);bs=transport_b(reference,model,brefs)
        for bref,b in zip(brefs,bs):
            full=statistics(d,c,b,model)
            effective=statistics(d,c,b,model,probability_fn=probability_effective)
            result["rows"].append(dict(detuning_mhz=model.detuning_mhz,name=row["name"],reference_b=float(bref),
                full_variance=full["variance"],effective_variance=effective["variance"],
                full_shot_variance=full["shot_variance"],effective_shot_variance=effective["shot_variance"]))
        closed=RamanD1(detuning_mhz=row["detuning_mhz"],emission=False)
        flags,phases,_,_=layout(d["parameters"],d["pattern"],c)
        rng=np.random.default_rng(67919);z=rng.normal(0,.015,(3,len(flags),2,6))
        b=transport_b(reference,closed,[-.6,0.,.6]);delta=np.array([-.0005,0.,.0005])
        exact=probability_d1(delta,b,z,d["parameters"],d["family"],flags,phases,c,model=closed)
        approx=probability_effective(delta,b,z,d["parameters"],d["family"],flags,phases,c,model=closed)
        result["coherent_checks"].append(dict(detuning_mhz=model.detuning_mhz,name=row["name"],
            max_population_difference=float(np.max(abs(exact-approx)))))
        (OUT/args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(detuning_mhz=model.detuning_mhz,name=row["name"],
            full_variance=float(np.mean([r["full_variance"] for r in result["rows"][-7:]])),
            effective_variance=float(np.mean([r["effective_variance"] for r in result["rows"][-7:]])),
            coherent_check=result["coherent_checks"][-1])),flush=True)
        model.pulse_map.cache_clear()


if __name__=="__main__":main()
