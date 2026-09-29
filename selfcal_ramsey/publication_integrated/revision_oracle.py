"""Counterfactual branch outcomes with shared calibration and OU trajectories."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from revision_study import estimators, DATA
from conditional_selector import Selector
from study import paired_noise, select, layout, atom_counts, probability_d1, RamanD1


FILE = DATA / "resolved_d1_oracle.json"


def run():
    est = estimators()
    branches = [est[f"branch_{j}"] for j in range(8)]
    mc = Selector()
    if FILE.exists():
        out=json.loads(FILE.read_text())
    else:
        out=dict(decay_resolution="excited_hyperfine",substeps=1,delta=.0005,reps=48,
                 b_values=[-.6,-.4,-.2,0.,.2,.4,.6],rows=[])
    for i,b in enumerate(out["b_values"][len(out["rows"]):],len(out["rows"])):
        delta=out["delta"];reps=out["reps"]
        rng=np.random.default_rng(259260+i*101)
        entry=branches[0].entry;d,c=entry["design"],entry["config"]
        noise=paired_noise([entry],reps,rng,1,c)["branch_0"]
        uniform=rng.uniform(1e-12,1-1e-12,(reps,c["shots"]))
        flags,phases=layout(d["parameters"],d["pattern"],c)[:2]
        counts=atom_counts(d["parameters"],d["family"],flags,c)
        short=dict(c,frequency_offsets=c["frequency_offsets"][:2],
                   segment_phase_offsets=c["segment_phase_offsets"][:2])
        model=RamanD1(decay_resolution="excited_hyperfine")
        pc=probability_d1(delta,b,noise[:,:2],d["parameters"],d["family"],
                          flags[:2],phases[:2],short,1,model)
        ycal=binom.ppf(uniform[:,:2],counts[:2],pc)/counts[:2]
        choices_local=select(branches,ycal)
        choices_mc=mc.select(ycal)
        losses=np.empty((reps,8))
        for j,e in enumerate(branches):
            p=probability_d1(delta,b,noise,d["parameters"],d["family"],
                             flags,phases,e.entry["config"],1,model)
            data=np.column_stack((ycal,binom.ppf(uniform[:,2:],counts[2:],p[:,2:])/counts[2:]))
            fit=e.fit(data)
            losses[:,j]=(fit[:,0]-delta)**2
            print(json.dumps(dict(b=b,branch=j,mse_hz2=float(losses[:,j].mean()*1e8))),flush=True)
        out["rows"].append(dict(b=b,seed=259260+i*101,losses=losses.tolist(),
                                choices_local=choices_local.tolist(),choices_mc=choices_mc.tolist()))
        FILE.write_text(json.dumps(out,indent=2))


if __name__=="__main__":run()
