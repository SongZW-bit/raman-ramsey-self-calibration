"""Coupled one-versus-three OU substeps for resolved-D1 endpoint risks."""
import json

import numpy as np
from scipy.stats import binom

from revision_study import estimators, DATA
from study import paired_noise, select, layout, atom_counts, probability_d1, RamanD1


def run():
    est=estimators();model=RamanD1(decay_resolution="excited_hyperfine");rows=[]
    for bi,b in enumerate((-.6,.6)):
        rng=np.random.default_rng(259280+bi)
        entries=[est[n].entry for n in ("fixed_refined","branch_0")]
        noise=paired_noise(entries,48,rng,3,entries[0]["config"])
        uniform=rng.uniform(1e-12,1-1e-12,(48,12))
        for steps in (1,3):
            for arm in ("fixed_refined","adaptive8"):
                if arm=="fixed_refined":
                    e=est[arm];d,c=e.entry["design"],e.entry["config"]
                    flags,phases=layout(d["parameters"],d["pattern"],c)[:2]
                    counts=atom_counts(d["parameters"],d["family"],flags,c)
                    nn=noise[arm]
                    if steps==1:nn=nn.reshape(48,12,2,6,3)[...,1].reshape(48,12,2,6)
                    p=probability_d1(.0005,b,nn,d["parameters"],d["family"],flags,phases,c,steps,model)
                    fit=e.fit(binom.ppf(uniform,counts,p)/counts);choices=None
                else:
                    branches=[est[f"branch_{j}"] for j in range(8)]
                    e=branches[0];d,c=e.entry["design"],e.entry["config"]
                    flags,phases=layout(d["parameters"],d["pattern"],c)[:2]
                    counts=atom_counts(d["parameters"],d["family"],flags,c)
                    nn=noise["branch_0"]
                    if steps==1:nn=nn.reshape(48,12,2,6,3)[...,1].reshape(48,12,2,6)
                    short=dict(c,frequency_offsets=c["frequency_offsets"][:2],
                               segment_phase_offsets=c["segment_phase_offsets"][:2])
                    pc=probability_d1(.0005,b,nn[:,:2],d["parameters"],d["family"],
                                      flags[:2],phases[:2],short,steps,model)
                    yc=binom.ppf(uniform[:,:2],counts[:2],pc)/counts[:2]
                    choices=select(branches,yc);fit=np.empty((48,2))
                    for j in np.unique(choices):
                        ids=np.flatnonzero(choices==j);ee=branches[j]
                        p=probability_d1(.0005,b,nn[ids],d["parameters"],d["family"],
                                         flags,phases,ee.entry["config"],steps,model)
                        y=np.column_stack((yc[ids],binom.ppf(uniform[ids,2:],counts[2:],p[:,2:])/counts[2:]))
                        fit[ids]=ee.fit(y)
                row=dict(b=b,arm=arm,steps=steps,estimates=fit.tolist(),
                         choices=None if choices is None else choices.tolist(),
                         mse_hz2=float(np.mean((fit[:,0]-.0005)**2)*1e8))
                rows.append(row)
                (DATA/"resolved_d1_convergence.json").write_text(json.dumps(rows,indent=2))
                print(json.dumps({k:v for k,v in row.items() if k not in ("estimates","choices")}),flush=True)


if __name__=="__main__":run()
