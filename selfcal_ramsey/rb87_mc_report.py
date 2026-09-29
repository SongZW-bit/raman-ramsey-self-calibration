"""Aggregate finite-count risk with stratified paired bootstrap intervals."""
import argparse
import json
import numpy as np
from rb87_multilevel_pilot import OUT


def summarize_adaptive(source):
    names=sorted(set(r["name"] for r in source["rows"]))
    points=sorted(set(r["b"] for r in source["rows"]))
    errors={}
    for name in names:
        rows=[next(r for r in source["rows"] if r["name"]==name and r["b"]==b) for b in points]
        errors[name]=np.array([(np.array(r["estimates"])[:,0]-source["true_delta"])**2 for r in rows])
    rng=np.random.default_rng(40512);boot={name:[] for name in names}
    for _ in range(5000):
        ix=rng.integers(0,source["reps"],size=(len(points),source["reps"]))
        for name in names:
            local_ix=ix if source.get("paired_randomness",False) or name in ("adaptive","fixed_bank") else rng.integers(0,source["reps"],size=ix.shape)
            boot[name].append(float(np.take_along_axis(errors[name],local_ix,axis=1).mean()))
    result=dict(effective=source.get("effective",False),points=points,reps_per_point=source["reps"],
        means={name:float(error.mean()) for name,error in errors.items()},comparisons=[])
    for name in names:
        if name=="adaptive":continue
        gain=1-errors["adaptive"].mean()/errors[name].mean()
        interval=np.quantile(1-np.array(boot["adaptive"])/np.array(boot[name]),[.025,.975])
        result["comparisons"].append(dict(reference=name,mse_reduction=float(gain),bootstrap_95_interval=interval.tolist()))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();source=json.loads((OUT/args.source).read_text());result=summarize_adaptive(source)
    result.update(source=args.source,scope="Stratified bootstrap Monte Carlo uncertainty for these frozen candidates and test points; not literature-wide superiority or a global optimum.")
    (OUT/args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
