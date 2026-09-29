"""Exact-atom finite-count check of frozen joint-design candidates.

Comparisons use the same physical unknown field ratio, atom budget, and
optical resource caps. Fixed designs all infer target frequency and the
unknown ratio; no oracle nuisance value is supplied to inference.
"""
import argparse
import json
import time
import numpy as np
from rb87_multilevel import RamanD1,probability_d1
from rb87_adaptive_experiment import Estimator,training_row,OUT
from rb87_joint_design import transport_b
from two_beam_adaptive import trajectories
from two_beam import atom_counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["train","experiment","validate"])
    p.add_argument("--source",default="rb87_joint_phase_refined.json")
    p.add_argument("--lookup",default="rb87_joint_confirmation_lookup.json")
    p.add_argument("--output",default="rb87_joint_confirmation.json")
    p.add_argument("--reps",type=int,default=100)
    p.add_argument("--seed",type=int,default=901278)
    p.add_argument("--substeps",type=int,default=1)
    p.add_argument("--points",type=float,nargs="+",default=[-.6,.6])
    p.add_argument("--detunings",type=float,nargs="+",default=[200.,1000.])
    p.add_argument("--names",nargs="+",default=["fixed_hyper","fixed_shape"])
    p.add_argument("--delta",type=float,default=.0005)
    p.add_argument("--effective",action="store_true")
    args=p.parse_args();reference=RamanD1();path=OUT/args.lookup
    probability_fn=probability_d1
    if args.effective:
        from rb87_effective_compare import probability_effective
        probability_fn=probability_effective
    if args.mode=="validate":
        saved=json.loads(path.read_text());checks=[]
        assert saved.get("effective",False)==args.effective
        from drift_study import layout
        for e in saved["entries"]:
            model=RamanD1(detuning_mhz=e["model"]["detuning_mhz"])
            estimator=Estimator(e["entry"],e["grid"],e["rows"])
            d,c=e["entry"]["design"],e["entry"]["config"]
            flags,phases,_,_=layout(d["parameters"],d["pattern"],c)
            b=transport_b(reference,model,[-.61,-.43,.13,.47,.61])
            delta=np.array([.0005,-.0005,.0011,-.0023,.0027])
            exact=probability_fn(delta,b,np.zeros((5,len(flags),2,6)),d["parameters"],d["family"],flags,phases,c,model=model)
            error=estimator.mean(delta,b,False)-exact
            white=np.einsum("ni,nij,nj->n",error,np.linalg.inv(estimator.cov(b)),error)
            mineig=float(np.linalg.eigvalsh(estimator.cov(np.linspace(e["grid"][0],e["grid"][-1],401))).min())
            checks.append(dict(key=e["key"],max_population_error=float(abs(error).max()),max_whitened_error=float(white.max()),minimum_covariance_eigenvalue=mineig))
        validation=dict(checks=checks,passed=all(c["max_whitened_error"]<.01 and c["minimum_covariance_eigenvalue"]>0 for c in checks))
        (OUT/args.output).write_text(json.dumps(validation,indent=2));print(json.dumps(validation,indent=2),flush=True)
        return
    if args.mode=="train":
        source=json.loads((OUT/args.source).read_text())
        entries=[r for r in source["rows"] if r["detuning_mhz"] in args.detunings and r["name"] in args.names]
        saved=dict(source=args.source,effective=args.effective,entries=[])
        if path.exists():saved=json.loads(path.read_text());assert saved["source"]==args.source
        assert saved.get("effective",False)==args.effective
        for row in entries:
            model=RamanD1(detuning_mhz=row["detuning_mhz"])
            key=str(row["detuning_mhz"])+"_"+row["name"]
            grid=np.sort(transport_b(reference,model,np.linspace(-.8,.8,25)))
            matches=[e for e in saved["entries"] if e["key"]==key]
            if matches:
                e=matches[0];assert e["entry"]==row["entry"] and np.array_equal(grid,e["grid"])
            else:
                e=dict(key=key,model=model.metadata(),entry=row["entry"],grid=grid.tolist(),rows=[])
                saved["entries"].append(e)
            for b in grid[len(e["rows"]):]:
                e["rows"].append(training_row(row["entry"],b,model,probability_fn=probability_fn))
                path.write_text(json.dumps(saved,indent=2))
                print(json.dumps(dict(stage="training",key=key,completed=len(e["rows"]),total=len(grid))),flush=True)
            model.pulse_map.cache_clear()
        return
    saved=json.loads(path.read_text())
    assert saved.get("effective",False)==args.effective
    result=dict(seed=args.seed,reps=args.reps,true_delta=args.delta,effective=args.effective,
        substeps=args.substeps,source=args.source,lookup=args.lookup,
        scope="OU trajectory and binomial data from the stated model (full D1 or matched coherent qubit), with Gaussian marginal inference. Frozen local-risk optimized candidates; independent Monte Carlo seed. No adaptive-selection claim.",rows=[])
    for e in saved["entries"]:
        if e["model"]["detuning_mhz"] not in args.detunings or e["entry"]["name"] not in args.names:continue
        model=RamanD1(detuning_mhz=e["model"]["detuning_mhz"])
        assert len(e["rows"])==len(e["grid"])
        estimator=Estimator(e["entry"],e["grid"],e["rows"])
        d,c=e["entry"]["design"],e["entry"]["config"]
        for bref in args.points:
            start=time.perf_counter();b=transport_b(reference,model,[bref])[0]
            rng=np.random.default_rng(args.seed+round((bref+.8)*10000))
            noise,flags,phases=trajectories(d,c,b,args.reps,rng,args.delta,substeps=args.substeps)
            counts=atom_counts(d["parameters"],d["family"],flags,c)
            prob=probability_fn(args.delta,b,noise,d["parameters"],d["family"],flags,phases,c,args.substeps,model)
            data=rng.binomial(counts,prob)/counts
            fits=estimator.fit(data);errors=(fits[:,0]-args.delta)**2
            predicted=np.array(estimator.cov(b));residual=data-estimator.mean(args.delta,b)
            row=dict(key=e["key"],reference_b=bref,physical_b=float(b),mse=float(errors.mean()),
                mse_se=float(errors.std(ddof=1)/np.sqrt(args.reps)),bias=float(np.mean(fits[:,0]-args.delta)),
                boundary_fraction=float(np.mean((abs(fits[:,0])>.00349)|(fits[:,1]<e["grid"][0]+1e-6)|(fits[:,1]>e["grid"][-1]-1e-6))),
                known_parameter_whitened_residual=float(np.mean(np.einsum("ni,ij,nj->n",residual,np.linalg.inv(predicted),residual))),
                estimates=fits.tolist(),elapsed=time.perf_counter()-start)
            result["rows"].append(row);(OUT/args.output).write_text(json.dumps(result,indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!="estimates"}),flush=True)
        model.pulse_map.cache_clear()


if __name__=="__main__":main()
