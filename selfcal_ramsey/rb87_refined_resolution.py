"""Nested OU trajectories test pulse-time discretization of frozen controls."""
import json
import numpy as np
from rb87_multilevel import RamanD1,probability_d1
from rb87_adaptive_experiment import Estimator
from rb87_multilevel_pilot import OUT
from two_beam_adaptive import trajectories


def main():
    saved=json.loads((OUT/"rb87_refined_lookup.json").read_text());model=RamanD1();rows=[]
    for name,b in [("fixed_shape",-.6),("branch_1",-.6),("branch_7",.6)]:
        e=next(e for e in saved["designs"] if e["name"]==name)
        d,c=e["design"],e["config"];est=Estimator(e,saved["grid"],saved["tables"][name])
        fine,flags,phases=trajectories(d,c,b,4,np.random.default_rng(51184),.0005,substeps=9)
        blocks=fine.reshape(4,len(flags),2,6,9)
        probs={}
        for steps,ids in [(1,[4]),(3,[1,4,7]),(9,list(range(9)))]:
            noise=blocks[...,ids].reshape(4,len(flags),2,6*steps)
            probs[steps]=probability_d1(.0005,b,noise,d["parameters"],d["family"],flags,phases,c,steps,model)
        error=probs[1]-probs[9];white=np.einsum("ni,ij,nj->n",error,np.linalg.inv(est.cov(b)),error)
        error3=probs[3]-probs[9];white3=np.einsum("ni,ij,nj->n",error3,np.linalg.inv(est.cov(b)),error3)
        rows.append(dict(name=name,b=b,max_population_difference_1_to_9=float(abs(error).max()),
            max_whitened_squared_difference_1_to_9=float(white.max()),
            max_whitened_squared_difference_3_to_9=float(white3.max())))
        print(json.dumps(rows[-1]),flush=True)
        (OUT/"rb87_refined_resolution.json").write_text(json.dumps(dict(rows=rows),indent=2))
    result=dict(rows=rows,passed=all(r["max_whitened_squared_difference_1_to_9"]<.02 for r in rows),
        scope="Four nested exact-atom OU trajectories per frozen design; trajectory quadrature audit, not Monte Carlo MSE confidence.")
    (OUT/"rb87_refined_resolution.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":main()
