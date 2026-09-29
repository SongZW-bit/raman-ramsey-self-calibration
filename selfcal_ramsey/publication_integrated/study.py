"""Publication extension: paired simulations, controls, and numerical audits."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from rb87_multilevel import RamanD1, probability_d1
from rb87_effective_compare import probability_effective
from rb87_adaptive_experiment import Estimator, training_row
from rb87_multilevel_pilot import statistics, optical_resources
from two_beam import geometry, atom_counts
from two_beam_adaptive import posterior
from drift_study import layout

OLD = ROOT.parent / "results"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def dump(name, obj):
    (DATA / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load(name):
    return json.loads((OLD / name).read_text())


def clean(entry):
    return dict(name=entry["name"], design={k: copy.deepcopy(entry["design"][k])
                for k in ("family", "parameters", "pattern")}, config=copy.deepcopy(entry["config"]))


def existing(effective):
    s = load("rb87_refined_effective_lookup.json" if effective else "rb87_refined_lookup.json")
    entries = {e["name"]: clean(e) for e in s["designs"]}
    est = {n: Estimator(e, s["grid"], s["tables"][n]) for n, e in entries.items()}
    return entries, est


def train(entries, effective, filename):
    path = DATA / filename
    out = json.loads(path.read_text()) if path.exists() else dict(effective=effective, grid=np.linspace(-.8,.8,21).tolist(), designs=entries, tables={})
    assert out["designs"] == entries
    model = RamanD1()
    fn = probability_effective if effective else probability_d1
    for e in entries:
        rows = out["tables"].setdefault(e["name"], [])
        for b in out["grid"][len(rows):]:
            rows.append(training_row(e, b, model, fn))
            dump(filename, out)
        print(json.dumps(dict(trained=e["name"], effective=effective)), flush=True)
    return {e["name"]: Estimator(e, out["grid"], out["tables"][e["name"]]) for e in entries}


def refine():
    entries, _ = existing(True)
    points = np.linspace(-.6,.6,7)
    model = RamanD1()
    candidates, history = [], []
    for family in ("fixed_shape", "fixed_hyper"):
        e = clean(entries[family]); v = e["design"]["parameters"]
        ids = [5,6,7,8,9,10,11] if family == "fixed_shape" else [5,6,7]
        x0 = np.array(v)[ids]
        freq_index = ids.index(10 if family == "fixed_shape" else 6)
        bounds = [(x-np.pi,x+np.pi) for x in x0]; bounds[freq_index]=(-.9,.9)
        for offset in (-.45, 0., .45):
            ee = clean(e); vv = ee["design"]["parameters"]
            best = [np.inf, None]; evaluations = [0]
            def objective(x):
                for j, y in zip(ids,x): vv[j] = float(y)
                loss = np.mean([statistics(ee["design"],ee["config"],b,model,
                                probability_fn=probability_effective)["variance"] for b in points])/1e-7
                evaluations[0] += 1
                if loss < best[0]: best[:] = [loss,np.array(x)]
                return loss
            initial=x0.copy();initial[freq_index]=offset
            fit=minimize(objective,initial,method="L-BFGS-B",bounds=bounds,
                         options=dict(maxiter=35,ftol=2e-7,eps=2e-5,maxls=12))
            for j,y in zip(ids,best[1]): vv[j]=float(y)
            ee["name"]=family+"_refined_"+str(len(candidates))
            candidates.append(ee)
            h=dict(name=ee["name"], effective_risk=best[0]*1e-7, success=bool(fit.success),
                   message=str(fit.message), evaluations=evaluations[0], start_frequency=offset)
            history.append(h);print(json.dumps(h),flush=True)
            dump("fixed_search.json",dict(prior=points.tolist(),candidates=candidates,history=history))
    selected={}
    for effective in (True,False):
        fn=probability_effective if effective else probability_d1
        risks=[]
        for e in list(entries.values())[:2]+candidates:
            rr=[statistics(e["design"],e["config"],b,model,probability_fn=fn)["variance"] for b in points]
            risks.append(dict(name=e["name"],risk=float(np.mean(rr)),point_risks=rr))
            print(json.dumps(dict(stage="fixed_screen",effective=effective,**risks[-1])),flush=True)
        best=min(risks,key=lambda r:r["risk"])
        ee=next(clean(e) for e in list(entries.values())[:2]+candidates if e["name"]==best["name"])
        ee["name"]="fixed_refined"
        selected[str(effective)]=dict(entry=ee,screen=risks)
        dump("fixed_selected.json",selected)
        train([ee],effective,"fixed_lookup_"+("effective" if effective else "d1")+".json")


def ablation():
    entries,_=existing(True)
    base=entries["branch_0"]
    variants=[]
    for kind in ("frequency", "phase"):
        for name in ("branch_0","branch_1","branch_7"):
            e=clean(entries[name]);e["name"]=kind+"_"+name
            field="segment_phase_offsets" if kind=="frequency" else "frequency_offsets"
            e["config"][field]=copy.deepcopy(base["config"][field])
            variants.append(e)
    train(variants,True,"ablation_lookup.json")


def extended_bank(effective):
    source=load("rb87_adaptive_refined.json")
    variants=[clean(row["entry"]) for row in source["entries"] if row["name"] not in ("branch_0","branch_1","branch_7")]
    train(variants,effective,"extended_lookup_"+("effective" if effective else "d1")+".json")


def add_lookup(estimators, filename):
    p=DATA/filename
    if not p.exists():return
    s=json.loads(p.read_text())
    for e in s["designs"]:
        estimators[e["name"]]=Estimator(e,s["grid"],s["tables"][e["name"]])


def paired_noise(entries,reps,rng,steps,noise_config,delay=0.,nested=None):
    times=[];shapes=[]
    for entry in entries:
        d,c=entry["design"],entry["config"]
        flags=layout(d["parameters"],d["pattern"],c)[0]
        tt=geometry(d["parameters"],d["family"],flags,c,steps)[0]
        if delay and not entry["name"].startswith("fixed"):
            tt=tt.copy();tt[2:]+=delay
        times.append(tt.ravel());shapes.append(tt.shape)
    union=np.unique(np.concatenate(times));path=np.empty((reps,2,len(union)))
    c=noise_config
    std=c["sigma_intensity"]*np.sqrt(np.array([1+c["beam_correlation"],1-c["beam_correlation"]])/2)
    path[:,:,0]=rng.normal(size=(reps,2))*std
    for j in range(1,len(union)):
        rr=np.exp(-(union[j]-union[j-1])/c["correlation_time"])
        path[:,:,j]=rr*path[:,:,j-1]+np.sqrt(1-rr*rr)*rng.normal(size=(reps,2))*std
    return {e["name"]:path[:,:,np.searchsorted(union,t)].reshape(reps,2,*s).transpose(0,2,1,3)
            for e,t,s in zip(entries,times,shapes)}


def select(branches,ycal):
    grid=np.linspace(-.8,.8,161);dg=np.linspace(-.003,.003,21)
    prefix=branches[0]
    post=posterior(ycal,prefix.mean(0.,grid)[:,:2],prefix.cov(grid)[:,:2,:2],
                   prefix.coeff(grid)[:,1,:2]/.003,dg)
    risk=np.array([e.risk(grid) for e in branches])
    return np.argmin(post@risk.T,axis=1)


def run_cell(estimators,b,delta,reps,seed,effective,arms,steps=3,noise_override=None,delay=0.,keep_data=False,decay_resolution="unresolved_excited",selector_file=None):
    fn=probability_effective if effective else probability_d1
    model=RamanD1(decay_resolution=decay_resolution);rng=np.random.default_rng(seed)
    prefix=estimators["branch_0"];cfg=dict(prefix.entry["config"])
    if noise_override:cfg.update(noise_override)
    base_names=[n for n in arms if n.startswith("fixed") and n!="fixed_bank"]+["branch_0"]
    noise=paired_noise([estimators[n].entry for n in base_names],reps,rng,steps,cfg,delay)
    uniform=rng.uniform(1e-12,1-1e-12,(reps,cfg["shots"]))
    e=prefix;d,c=e.entry["design"],e.entry["config"]
    flags,phases=layout(d["parameters"],d["pattern"],c)[:2]
    counts=atom_counts(d["parameters"],d["family"],flags,c)
    short=dict(c,frequency_offsets=c["frequency_offsets"][:2],segment_phase_offsets=c["segment_phase_offsets"][:2])
    pcal=fn(delta,b,noise["branch_0"][:,:2],d["parameters"],d["family"],flags[:2],phases[:2],short,steps,model)
    ycal=binom.ppf(uniform[:,:2],counts[:2],pcal)/counts[:2]
    rows=[]
    for arm in arms:
        start=time.perf_counter()
        if arm.startswith("fixed") and arm!="fixed_bank":
            e=estimators[arm];dd,cc=e.entry["design"],e.entry["config"]
            ff,pp=layout(dd["parameters"],dd["pattern"],cc)[:2]
            count=atom_counts(dd["parameters"],dd["family"],ff,cc)
            p=fn(delta,b,noise[arm],dd["parameters"],dd["family"],ff,pp,cc,steps,model)
            data=binom.ppf(uniform,count,p)/count;fit=e.fit(data)
            choices=None
        else:
            names=["branch_0","branch_1","branch_7"]
            if arm in ("frequency","phase"):names=[arm+"_"+n for n in names]
            if arm in ("adaptive8","adaptive8_mc"):names=["branch_"+str(j) for j in range(8)]
            branches=[estimators[n] for n in names]
            if arm=="fixed_bank":
                # Same seven-point design prior for fixed-bank and refined arms.
                ri=np.array([e.risk(np.linspace(-.6,.6,7)).mean() for e in branches])
                choices=np.full(reps,np.argmin(ri),int)
            elif arm=="adaptive8_mc":
                from conditional_selector import Selector
                choices=Selector(selector_file).select(ycal)
            else: choices=select(branches,ycal)
            data=np.empty((reps,len(flags)));fit=np.empty((reps,2))
            for j in np.unique(choices):
                ids=np.flatnonzero(choices==j);e=branches[j]
                p=fn(delta,b,noise["branch_0"][ids],d["parameters"],d["family"],flags,phases,e.entry["config"],steps,model)
                y=np.column_stack((ycal[ids],binom.ppf(uniform[ids,2:],counts[2:],p[:,2:])/counts[2:]))
                data[ids]=y;fit[ids]=e.fit(y)
        errors=fit[:,0]-delta
        row=dict(name=arm,b=b,delta=delta,reps=reps,seed=seed,effective=effective,substeps=steps,
                 noise=cfg|{},delay=delay,estimates=fit.tolist(),mse=float(np.mean(errors**2)),
                 bias=float(np.mean(errors)),abs_error_q95=float(np.quantile(abs(errors),.95)),
                 abs_error_q99=float(np.quantile(abs(errors),.99)),
                 target_bound_hits=int(np.sum(abs(fit[:,0])>.003499)),
                 nuisance_bound_hits=int(np.sum(abs(fit[:,1])>.79999)),
                 choices=None if choices is None else choices.tolist(),seconds=time.perf_counter()-start)
        row["noise"]={k:cfg[k] for k in ("sigma_intensity","correlation_time","beam_correlation")}
        if keep_data:row["data"]=data.tolist()
        rows.append(row)
        print(json.dumps({k:row[k] for k in ("name","b","delta","mse","bias","target_bound_hits","seconds")}),flush=True)
    return rows


def simulate(suite):
    effective=suite not in ("d1","d1_stress","bank8_d1")
    _,est=existing(effective)
    add_lookup(est,"fixed_lookup_"+("effective" if effective else "d1")+".json")
    arms=["fixed_shape","fixed_hyper","fixed_bank","adaptive"]
    if "fixed_refined" in est:arms.insert(2,"fixed_refined")
    if suite=="broad":
        cells=[(b,d,200,{},0.) for b in [-.6,-.43,-.21,0.,.17,.39,.6] for d in [-.0015,.0005,.002]]
    elif suite=="d1":
        cells=[(b,d,64,{},0.) for b in [-.43,.17,.43] for d in [-.0015,.002]]
        arms=["fixed_shape","adaptive"]+(["fixed_refined"] if "fixed_refined" in est else [])
    elif suite=="ablations":
        add_lookup(est,"ablation_lookup.json")
        arms=["fixed_bank","frequency","phase","adaptive"]
        cells=[(b,.0005,500,{},0.) for b in [-.6,.6]]
    elif suite=="stress":
        scenarios=[({},0.),({"correlation_time":60.},0.),({"correlation_time":1500.},0.),
                   ({"sigma_intensity":.015},0.),({"sigma_intensity":.06},0.),
                   ({"beam_correlation":-.7},0.),({"beam_correlation":.7},0.),({},100.),({},300.)]
        cells=[(b,.0005,240,s,l) for s,l in scenarios for b in [-.6,.6]]
        arms=["fixed_shape","adaptive"]
    elif suite in ("bank8","bank8_d1"):
        add_lookup(est,"extended_lookup_"+("effective" if effective else "d1")+".json")
        arms=["fixed_shape","fixed_refined","adaptive","adaptive8"]
        if not effective:arms=["fixed_refined","adaptive8"]
        bs=[-.6,-.43,-.21,0.,.17,.39,.6] if effective else [-.6,-.43,.17,.43,.6]
        ds=[.0005] if effective else [.0005]
        cells=[(b,d,400 if effective else 96,{},0.) for b in bs for d in ds]
    elif suite=="d1_stress":
        cells=[(b,.0005,64,s,l) for s,l in [({"sigma_intensity":.06},0.),({},100.)] for b in [-.6,.6]]
        arms=["fixed_shape","adaptive"]
    else:raise ValueError(suite)
    file=suite+".json"
    result=json.loads((DATA/file).read_text()) if (DATA/file).exists() else dict(suite=suite,arms=arms,rows=[],completed_cells=0,
            scope="Frozen designs; nominal approximate likelihood even in noise mismatch and latency tests; paired exact nonlinear trajectory generation.")
    seed0={"broad":910001,"d1":930001,"ablations":950001,"stress":970001,"d1_stress":990001,"bank8":141001,"bank8_d1":151001}[suite]
    for i,(b,d,n,s,l) in enumerate(cells):
        if i<result["completed_cells"]:continue
        result["rows"]+=run_cell(est,b,d,n,seed0+i*103,effective,arms,noise_override=s,delay=l,keep_data=True)
        result["completed_cells"]=i+1;dump(file,result)


def convergence(effective):
    _,est=existing(effective);fn=probability_effective if effective else probability_d1
    model=RamanD1();out=dict(effective=effective,rows=[],reps=96,seed=111001)
    for bi,b in enumerate([-.6,.6]):
        rng=np.random.default_rng(111001+bi)
        entries=[est[n].entry for n in ("fixed_shape","branch_0")]
        noises=paired_noise(entries,96,rng,9,entries[0]["config"])
        uniforms=rng.uniform(1e-12,1-1e-12,(96,12))
        for steps,indices in [(3,[1,4,7]),(9,list(range(9)))]:
            for name in ("fixed_shape","adaptive"):
                e=est["fixed_shape"] if name=="fixed_shape" else est["branch_0"]
                d,c=e.entry["design"],e.entry["config"];f,p=layout(d["parameters"],d["pattern"],c)[:2]
                count=atom_counts(d["parameters"],d["family"],f,c)
                nn=noises[e.entry["name"]].reshape(96,12,2,6,9)[...,indices].reshape(96,12,2,6*steps)
                if name=="fixed_shape":
                    prob=fn(.0005,b,nn,d["parameters"],d["family"],f,p,c,steps,model)
                    fits=e.fit(binom.ppf(uniforms,count,prob)/count);choices=None
                else:
                    short=dict(c,frequency_offsets=c["frequency_offsets"][:2],segment_phase_offsets=c["segment_phase_offsets"][:2])
                    pc=fn(.0005,b,nn[:,:2],d["parameters"],d["family"],f[:2],p[:2],short,steps,model)
                    yc=binom.ppf(uniforms[:,:2],count[:2],pc)/count[:2]
                    branches=[est[n] for n in ("branch_0","branch_1","branch_7")]
                    choices=select(branches,yc);fits=np.empty((96,2))
                    for j in np.unique(choices):
                        ids=np.flatnonzero(choices==j);ee=branches[j]
                        prob=fn(.0005,b,nn[ids],d["parameters"],d["family"],f,p,ee.entry["config"],steps,model)
                        data=np.column_stack((yc[ids],binom.ppf(uniforms[ids,2:],count[2:],prob[:,2:])/count[2:]))
                        fits[ids]=ee.fit(data)
                row=dict(b=b,name=name,steps=steps,mse=float(np.mean((fits[:,0]-.0005)**2)),estimates=fits.tolist(),choices=None if choices is None else choices.tolist())
                out["rows"].append(row);dump("convergence_"+("effective" if effective else "d1")+".json",out)
                print(json.dumps({k:v for k,v in row.items() if k not in ("estimates","choices")}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("task",choices=["refine","train_ablation","train_extended","train_extended_d1","bank8","bank8_d1","broad","d1","ablations","stress","d1_stress","convergence","convergence_d1"])
    a=p.parse_args()
    if a.task=="refine":refine()
    elif a.task=="train_ablation":ablation()
    elif a.task.startswith("train_extended"):extended_bank(a.task=="train_extended")
    elif a.task.startswith("convergence"):convergence(a.task=="convergence")
    else:simulate(a.task)


if __name__=="__main__":main()
