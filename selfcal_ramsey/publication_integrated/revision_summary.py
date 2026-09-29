"""Paired risk intervals for the resolved-D1 follow-up ensembles."""
import json

import numpy as np

from study import DATA


def paired(rows,reference,adaptive,hierarchical=False,seed=259270,bootstrap=6000):
    keys=sorted({(r["b"],r["delta"],r["seed"]) for r in rows})
    rng=np.random.default_rng(seed)
    means=np.zeros((len(keys),2))
    draws=np.zeros((bootstrap,len(keys),2))
    for i,key in enumerate(keys):
        pair=[]
        for name in (reference,adaptive):
            row=next(r for r in rows if (r["b"],r["delta"],r["seed"],r["name"])==(*key,name))
            pair.append((np.asarray(row["estimates"])[:,0]-row["delta"])**2)
        loss=np.column_stack(pair)
        means[i]=loss.mean(axis=0)
        index=rng.integers(0,len(loss),(bootstrap,len(loss)))
        draws[:,i,:]=loss[index].mean(axis=1)
    if hierarchical:
        cells=rng.integers(0,len(keys),(bootstrap,len(keys)))
        boot=np.take_along_axis(draws,cells[:,:,None],axis=1).mean(axis=1)
    else:
        boot=draws.mean(axis=1)
    risk=means.mean(axis=0)
    gain=1-risk[1]/risk[0]
    return dict(cells=len(keys),records_per_cell=len(loss),reference=reference,adaptive=adaptive,
                reference_mse_hz2=float(risk[0]*1e8),adaptive_mse_hz2=float(risk[1]*1e8),
                gain=float(gain),ci=np.quantile(1-boot[:,1]/boot[:,0],[.025,.975]).tolist(),
                hierarchical=hierarchical)


def main():
    out={}
    for source,arms,hierarchical in [
        ("resolved_d1_endpoints.json",[("fixed_shape","adaptive8"),("fixed_refined","adaptive8")],False),
        ("resolved_d1_holdout.json",[("fixed_shape","adaptive8"),("fixed_refined","adaptive8"),
                                     ("fixed_refined","adaptive8_mc"),("adaptive8","adaptive8_mc")],True)]:
        path=DATA/source
        if not path.exists():continue
        s=json.loads(path.read_text());rows=s["rows"]
        if s["completed_cells"]<len(s["cells"]):continue
        out[source]={"comparisons":[paired(rows,*arm,hierarchical=hierarchical) for arm in arms]}
        if "holdout" in source:
            out[source]["by_region"]={}
            for label,low,high in [("negative",-.6,-.2),("interior",-.2,.2),("positive",.2,.6)]:
                part=[r for r in rows if low<=r["b"]<(high if high<.6 else .60000001)]
                out[source]["by_region"][label]=paired(part,"fixed_refined","adaptive8",True)
            out[source]["selection_frequencies"]={arm:np.bincount(
                np.concatenate([r["choices"] for r in rows if r["name"]==arm]),minlength=8).tolist()
                for arm in ("adaptive8","adaptive8_mc")}
    source=DATA/"resolved_d1_oracle.json"
    if source.exists():
        s=json.loads(source.read_text());out["oracle"]=[]
        for row in s["rows"]:
            loss=np.asarray(row["losses"])
            best=int(np.argmin(loss.mean(axis=0)))
            local=np.asarray(row["choices_local"]);mc=np.asarray(row["choices_mc"])
            out["oracle"].append(dict(b=row["b"],best_parameter_branch=best,
                branch_risk_hz2=(loss.mean(axis=0)*1e8).tolist(),
                local_choices=np.bincount(local,minlength=8).tolist(),
                mc_choices=np.bincount(mc,minlength=8).tolist(),
                local_misselect=float(np.mean(local!=best)),mc_misselect=float(np.mean(mc!=best)),
                parameter_oracle_hz2=float(loss[:,best].mean()*1e8),
                realization_oracle_hz2=float(loss.min(axis=1).mean()*1e8),
                local_hz2=float(loss[np.arange(len(loss)),local].mean()*1e8),
                mc_hz2=float(loss[np.arange(len(loss)),mc].mean()*1e8)))
    path=DATA/"resolved_200_mc.json"
    if path.exists():
        s=json.loads(path.read_text());out["detuning_200"]={}
        for key in sorted({r["key"] for r in s["rows"]}):
            rows=[r for r in s["rows"] if r["key"]==key]
            out["detuning_200"][key]=dict(cells=len(rows),mse_hz2=float(np.mean([r["mse"] for r in rows])*1e8),
                boundary_fraction=float(np.mean([r["boundary_fraction"] for r in rows])))
    (DATA/"revision_summary.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()
