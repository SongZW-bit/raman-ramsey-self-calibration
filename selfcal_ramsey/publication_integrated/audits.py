"""Independent moment, optimization, theory, and literature diagnostics."""
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.linalg import expm
from scipy.stats import skew
from numpy.polynomial.hermite import hermgauss
from study import ROOT, DATA, OLD, dump, existing, add_lookup, RamanD1, probability_d1, probability_effective
from two_beam_adaptive import trajectories
from two_beam import atom_counts
from drift_study import layout


def moments(effective=True):
    _,est=existing(effective);model=RamanD1()
    fn=probability_effective if effective else probability_d1
    n=12000 if effective else 192
    out=dict(effective=effective,independent_paths=n,rows=[],
             scope="Independent nonlinear moments; covariance errors include finite-trajectory sampling. Not an exact marginal-likelihood certificate.")
    for j,(name,b,delta) in enumerate([("fixed_shape",-.6,.0005),("branch_1",-.6,.0005),
                    ("branch_7",.6,.0005),("fixed_shape",.17,.002),("branch_0",.17,.002)]):
        e=est[name];d,c=e.entry["design"],e.entry["config"]
        noise,f,ph=trajectories(d,c,b,n,np.random.default_rng(121001+j),delta,substeps=3)
        p=fn(delta,b,noise,d["parameters"],d["family"],f,ph,c,3,model)
        count=atom_counts(d["parameters"],d["family"],f,c)
        mu=p.mean(axis=0);cov=np.cov(p,rowvar=False)+np.diag((p*(1-p)).mean(axis=0)/count)
        model_mu=e.mean(delta,b);model_cov=e.cov(b);prec=np.linalg.inv(model_cov)
        a=e.coeff(b)[1]/.003;g=(e.mean(delta,b+1e-5)-e.mean(delta,b-1e-5))/2e-5
        jac=np.column_stack((a,g));w=np.linalg.solve(jac.T@prec@jac,jac.T@prec)[0]
        error=mu-model_mu
        target_noise=(p-model_mu)@w
        row=dict(name=name,b=b,delta=delta,max_mean_error=float(abs(error).max()),
                 whitened_mean_squared=float(error@prec@error),
                 target_bias_over_sd=float(w@error/np.sqrt(w@model_cov@w)),
                 target_variance_ratio=float(w@cov@w/(w@model_cov@w)),
                 drift_projection_skew=float(skew(target_noise)),
                 mean=mu.tolist(),covariance=cov.tolist())
        out["rows"].append(row);dump("moments_"+("effective" if effective else "d1")+".json",out)
        print(json.dumps({k:v for k,v in row.items() if k not in ("mean","covariance")}),flush=True)


def global_audit():
    _,est=existing(True);add_lookup(est,"fixed_lookup_effective.json")
    source=json.loads((DATA/"broad.json").read_text())
    out=dict(grid_shape=[81,401],rows=[],scope="Dense two-dimensional Gaussian-objective search plus 12 continuous starts; checks optimization, not correctness of the likelihood.")
    for row in source["rows"]:
        if row["name"] not in ("fixed_shape","fixed_refined","adaptive"):continue
        if row["b"] not in (-.6,.17,.6) or row["delta"]!=.0005:continue
        data=np.array(row["data"]);old=np.array(row["estimates"])
        # Include largest-error records as well as uniformly selected records.
        ids=np.unique(np.r_[np.arange(0,len(data),40),np.argsort(abs(old[:,0]-.0005))[-3:]])
        for i in ids:
            name=row["name"]
            if name=="adaptive":name=["branch_0","branch_1","branch_7"][row["choices"][i]]
            e=est[name];y=data[i]
            bg=np.linspace(-.8,.8,401);dg=np.linspace(-.0035,.0035,81)
            cov=e.cov(bg);prec=np.linalg.inv(cov);ld=np.linalg.slogdet(cov)[1]
            mu=np.array([e.mean(dd,bg) for dd in dg]);r=y-mu
            losses=np.einsum("dbi,bij,dbj->db",r,prec,r)+ld
            ordering=np.argsort(losses.ravel());starts=[]
            for flat in ordering:
                a,bidx=np.unravel_index(flat,losses.shape)
                if all(abs(bg[bidx]-ss[1])>.08 or abs(dg[a]-ss[0])>.0002 for ss in starts):
                    starts.append([dg[a],bg[bidx]])
                if len(starts)==12:break
            def obj(x):
                c=e.cov(x[1]);r=y-e.mean(x[0]*.001,x[1])
                return float(np.linalg.slogdet(c)[1]+r@np.linalg.solve(c,r))
            candidates=[minimize(obj,[x[0]*1000,x[1]],method="L-BFGS-B",bounds=[(-3.5,3.5),(-.8,.8)],
                                 options=dict(ftol=1e-12,maxiter=160)) for x in starts]
            best=min(candidates,key=lambda rr:rr.fun);oldloss=obj([old[i,0]*1000,old[i,1]])
            out["rows"].append(dict(arm=row["name"],b=row["b"],record=int(i),
                        objective_improvement=float(oldloss-best.fun),
                        frequency_difference=float(best.x[0]*.001-old[i,0]),success=bool(best.success)))
        dump("global_audit.json",out)
        print(json.dumps(dict(arm=row["name"],b=row["b"],records=len(ids))),flush=True)
    out["max_objective_improvement"]=max(r["objective_improvement"] for r in out["rows"])
    out["max_frequency_difference"]=max(abs(r["frequency_difference"]) for r in out["rows"])
    dump("global_audit.json",out)


def theory():
    _,estimators=existing(False)
    saved=json.loads((OLD/"rb87_refined_lookup.json").read_text())
    out=dict(geometry=[],ou=[],identity=[],timing=[])
    for name in ["fixed_shape","branch_0","branch_1","branch_7"]:
        for row,b in zip(saved["tables"][name],saved["grid"]):
            if abs(b)>.65:continue
            j=np.array(row["jacobian"]);cov=np.array(row["covariance"])
            fi=j.T@np.linalg.solve(cov,j);a=fi[0,0];ret=1-fi[0,1]**2/(fi[0,0]*fi[1,1])
            reconstructed=1/(a*ret)
            out["geometry"].append(dict(name=name,b=b,A=float(a),retention=float(ret),variance=reconstructed,
                        shot=row["shot_variance"],common=row["common_variance"],difference=row["difference_variance"],
                        relative_identity_error=float(abs(reconstructed/row["variance"]-1))))
    rng=np.random.default_rng(131001);n=200000;s2=.03**2/2;R=.4*s2
    z=rng.normal(0,np.sqrt(s2),n);y=z+rng.normal(0,np.sqrt(R),n)
    posterior_mean=s2/(s2+R)*y;p0=s2*R/(s2+R)
    for age in [0.,.1,.25,.5,1.,2.,4.]:
        a=np.exp(-age);future=a*z+np.sqrt(s2*(1-a*a))*rng.normal(size=n)
        correction=a*posterior_mean
        observed=float(np.mean(future**2-(future-correction)**2))
        predicted=a*a*(s2-p0)
        out["ou"].append(dict(age_over_tau=age,predicted_gain=predicted,observed_gain=observed,
                    standard_error=float(np.std(future**2-(future-correction)**2,ddof=1)/np.sqrt(n)),
                    conditional_variance=p0*a*a+s2*(1-a*a)))
    # Measure only the online posterior/lookup decision, after warm-up.
    from study import select
    branches=[estimators[n] for n in ("branch_0","branch_1","branch_7")]
    yy=branches[0].mean(.0005,-.6)[:2][None,:]
    select(branches,yy);times=[]
    for _ in range(200):
        t=time.perf_counter();select(branches,yy);times.append(time.perf_counter()-t)
    out["timing"]=dict(repetitions=200,median_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
             angular_unit_rad_per_second=2*np.pi*10000,
             scope="Python implementation on this workstation; hardware-independent switching latency is separately parameterized. No experimental latency measurement.")
    dump("theory.json",out)
    print(json.dumps(dict(max_information_identity_error=max(x["relative_identity_error"] for x in out["geometry"]),timing=out["timing"])),flush=True)


def literature():
    # Independent Hilbert-space reconstruction of conventional Hyper-Ramsey.
    sx=np.array([[0.,1.],[1.,0.]])
    sy=np.array([[0.,-1j],[1j,0.]])
    sz=np.diag([1.,-1.]);initial=np.array([1.,0.],complex)
    def p(delta,residual,phase,scale=1.,intensity=1.):
        omega=np.sqrt(intensity)*scale
        tau=np.pi/2;T=20.
        h=lambda ph:(omega*np.cos(ph)*sx+omega*np.sin(ph)*sy+(delta+residual)*sz)/2
        state=expm(-1j*h(0)*tau)@initial
        state=expm(-1j*delta*sz*T/2)@state
        state=expm(-1j*h(np.pi+phase)*2*tau)@state
        state=expm(-1j*h(phase)*tau)@state
        return abs(state[1])**2
    rows=[]
    for shift in np.geomspace(.003,.06,12):
        lock=brentq(lambda d:p(d,shift,np.pi/2)-p(d,shift,-np.pi/2),-.01,.01,xtol=1e-14)
        rows.append(dict(residual_shift=float(shift),locked_delta=float(lock)))
    slope=np.polyfit(np.log([r["residual_shift"] for r in rows]),np.log(abs(np.array([r["locked_delta"] for r in rows]))),1)[0]
    nodes,weights=hermgauss(48);weights/=np.sqrt(np.pi)
    fluctuations=[]
    for sigma in [0.,.01,.03,.06]:
        intensities=1+np.sqrt(2)*sigma*nodes
        def signal(delta):
            return sum(w*(p(delta,.5*(ii-1),np.pi/2,intensity=ii)-p(delta,.5*(ii-1),-np.pi/2,intensity=ii)) for ii,w in zip(intensities,weights))
        lock=brentq(signal,-.01,.01,xtol=1e-14)
        fluctuations.append(dict(fractional_intensity_sigma=sigma,locked_delta=float(lock)))
    out=dict(hyper_ramsey=rows,cubic_log_slope=float(slope),intensity_fluctuations=fluctuations,
             sources=["10.1103/PhysRevA.82.011804","10.1103/PhysRevA.97.031406"],
             scope="Independent equation-level reconstruction: cubic residual-shift suppression and correlated Rabi/light-shift intensity fluctuations. Not digitization or a reproduction of all published figure parameters; not a Raman performance baseline.")
    assert 2.95<slope<3.05
    dump("literature_checks.json",out);print(json.dumps(out),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("task",choices=["moments","moments_d1","global","theory","literature"]);a=p.parse_args()
    if a.task.startswith("moments"):moments(a.task=="moments")
    elif a.task=="global":global_audit()
    elif a.task=="theory":theory()
    else:literature()
