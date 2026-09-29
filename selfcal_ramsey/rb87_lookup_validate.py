"""Independent interpolation and time-sampling audits for a saved D1 lookup."""
import argparse
import json
import numpy as np
from rb87_adaptive_experiment import Estimator,OUT
from rb87_multilevel import RamanD1,probability_d1
from drift_study import layout
from two_beam_adaptive import trajectories


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lookup",default="rb87_d1_lookup.json")
    p.add_argument("--output",default="rb87_d1_lookup_validation.json")
    args=p.parse_args()
    saved=json.loads((OUT/args.lookup).read_text());model=RamanD1(detuning_mhz=saved["model"]["detuning_mhz"])
    rows=[];rng=np.random.default_rng(64430)
    for entry in saved["designs"]:
        if len(saved["tables"].get(entry["name"],[]))!=len(saved["grid"]):continue
        e=Estimator(entry,saved["grid"],saved["tables"][entry["name"]])
        d,c=entry["design"],entry["config"];v=d["parameters"]
        flags,phases,_,_=layout(v,d["pattern"],c)
        bs=np.array([-.61,-.43,.13,.47,.61]);delta=np.array([.0005,-.0005,.0011,-.0023,.0027])
        exact=probability_d1(delta,bs,np.zeros((5,len(flags),2,6)),v,d["family"],flags,phases,c,model=model)
        interpolated=e.mean(delta,bs,False);errors=interpolated-exact
        whitened=np.einsum("ni,nij,nj->n",errors,np.linalg.inv(e.cov(bs)),errors)
        grid=np.linspace(-.8,.8,801);minimum=float(np.linalg.eigvalsh(e.cov(grid)).min())
        rows.append(dict(name=entry["name"],max_population_error=float(abs(errors).max()),
                         max_whitened_squared_error=float(whitened.max()),minimum_covariance_eigenvalue=minimum))
    result=dict(rows=rows,scope="Noise-free mean interpolation and covariance positivity only; not the accuracy of Gaussian intensity marginalization.")
    result["passed"]=len(rows)==len(saved["designs"]) and all(r["max_whitened_squared_error"]<.01 and r["minimum_covariance_eigenvalue"]>0 for r in rows)
    (OUT/args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
