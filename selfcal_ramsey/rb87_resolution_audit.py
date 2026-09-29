"""Convergence of D1 local information against OU pulse sampling and steps."""
import json
import time
from rb87_adaptive_experiment import make_designs
from rb87_multilevel import RamanD1
from rb87_multilevel_pilot import statistics,OUT


def main():
    model=RamanD1();entries=make_designs(model)
    rows=[]
    for name,b in [("fixed_shape",-.6),("branch_1",-.6),("branch_6",.6)]:
        e=next(e for e in entries if e["name"]==name)
        for substeps,step in [(1,2e-4),(1,1e-4),(3,2e-4),(9,2e-4)]:
            start=time.perf_counter()
            s=statistics(e["design"],e["config"],b,model,step=step,substeps=substeps)
            row=dict(name=name,b=b,substeps=substeps,finite_difference_step=step,
                     variance=s["variance"],elapsed=time.perf_counter()-start)
            rows.append(row);print(json.dumps(row),flush=True)
            (OUT/"rb87_resolution_audit.json").write_text(json.dumps(dict(rows=rows),indent=2))
    groups=[]
    for name in ["fixed_shape","branch_1","branch_6"]:
        r=[r for r in rows if r["name"]==name]
        groups.append(dict(name=name,step_relative_difference=abs(r[1]["variance"]/r[0]["variance"]-1),
            sampling_1_to_9=abs(r[0]["variance"]/r[3]["variance"]-1),
            sampling_3_to_9=abs(r[2]["variance"]/r[3]["variance"]-1)))
    result=dict(rows=rows,comparisons=groups,scope="Local linear-noise information convergence only, not nonlinear Monte Carlo convergence.")
    result["passed"]=all(r["step_relative_difference"]<.001 and r["sampling_1_to_9"]<.01 for r in groups)
    (OUT/"rb87_resolution_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["comparisons"]),flush=True)


if __name__=="__main__":main()
