"""Finite-count pilot with exact D1 trajectory generation and spline inference.

Training uses the D1 master equation, never effective-qubit populations.
The likelihood marginalizes intensity noise with a local Gaussian covariance;
this approximation is explicit and shared by fixed and adaptive arms.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import binom
from rb87_multilevel import RamanD1, probability_d1
from rb87_multilevel_pilot import physical_design,statistics,optical_resources
from two_beam_adaptive import branch_config,trajectories,posterior
from two_beam import atom_counts,geometry
from drift_study import layout

OUT=Path(__file__).resolve().parent/"results"


def make_designs(model):
    fixed=json.loads((OUT/"two_beam_fixed_band_mc.json").read_text())
    adaptive=json.loads((OUT/"two_beam_adaptive_phase_band.json").read_text())
    designs=[]
    for e in fixed["designs"]:
        d,c=physical_design(e["design"],e["config"],model)
        designs.append(dict(name="fixed_"+d["family"],design=d,config=c))
    d,c=physical_design(adaptive["design"],adaptive["config"],model)
    for i,control in enumerate(adaptive["offsets"]):
        designs.append(dict(name="branch_"+str(i),design=d,
                            config=branch_config(c,control,adaptive["calibration_shots"])))
    return designs


def training_row(entry,b,model,probability_fn=probability_d1):
    d,c=entry["design"],entry["config"]
    stat=statistics(d,c,b,model,probability_fn=probability_fn)
    v=np.asarray(d["parameters"]);flags,phases,_,_=layout(v,d["pattern"],c)
    # Fourth-order polynomial for target detuning; the two-dimensional model
    # is validated later against independent exact populations.
    delta_nodes=np.array([-.003,-.0015,0.,.0015,.003])
    pp=probability_fn(delta_nodes,b,np.zeros((5,len(flags),2,6)),v,d["family"],flags,phases,c,model=model)
    polynomial=np.polynomial.polynomial.polyfit(delta_nodes/.003,pp,4)
    # Second-order mean correction with the pulse-centre OU covariance.
    k=len(flags);gap=(v[0]+v[1])/2+np.where(flags,v[3],v[2])
    corr=np.exp(-gap/c["correlation_time"]);factors=np.zeros((k,4,4))
    for ch in range(2):
        std=c["sigma_intensity"]*np.sqrt((1+(-1)**ch*c["beam_correlation"])/2)
        j=2*ch;factors[:,j,j]=std;factors[:,j+1,j]=std*corr
        factors[:,j+1,j+1]=std*np.sqrt(1-corr*corr)
    z=np.zeros((9,k,2,6));step=.2
    for direction in range(4):
        for ch in range(2):
            for pulse in range(2):
                x=step*factors[:,2*ch+pulse,direction,None]
                z[1+2*direction,:,ch,3*pulse:3*pulse+3]=x
                z[2+2*direction,:,ch,3*pulse:3*pulse+3]=-x
    p=probability_fn(0.,b,z,v,d["family"],flags,phases,c,model=model)
    correction=.5*np.sum(p[1::2]+p[2::2]-2*p[0],axis=0)/step**2
    stat.update(mean_polynomial=polynomial.tolist(),mean_correction=correction.tolist())
    return stat


def covariance_interpolator(grid, values):
    spline = CubicSpline(grid, values)
    h = np.diff(grid)[:, None, None]
    c = spline.c
    controls = np.stack((c[3], c[3]+c[2]*h/3,
        c[3]+2*c[2]*h/3+c[1]*h*h/3,
        c[3]+c[2]*h+c[1]*h*h+c[0]*h*h*h), axis=1)

    # A matrix Bernstein curve is a convex combination of its controls.
    # Recursive subdivision certifies positivity throughout each interval.
    def certified(points, depth=0):
        if np.linalg.eigvalsh(points).min() > 1e-14:
            return True
        if depth == 12:
            return False
        a = (points[:-1]+points[1:])/2
        b = (a[:-1]+a[1:])/2
        midpoint = (b[0]+b[1])/2
        return certified(np.array([points[0], a[0], b[0], midpoint]), depth+1) and certified(
            np.array([midpoint, b[1], a[2], points[3]]), depth+1)

    if all(certified(points) for points in controls):
        spline.interpolation_method = "certified_positive_cubic"
        return spline
    factors = np.linalg.cholesky(values)
    diagonal = np.arange(factors.shape[-1])
    factors[:, diagonal, diagonal] = np.log(factors[:, diagonal, diagonal])
    factor_spline = CubicSpline(grid, factors)

    def positive_covariance(b):
        factor = factor_spline(b)
        factor[..., diagonal, diagonal] = np.exp(factor[..., diagonal, diagonal])
        return factor @ np.swapaxes(factor, -1, -2)

    positive_covariance.interpolation_method = "log_cholesky_cubic"
    return positive_covariance


class Estimator:
    def __init__(self,entry,grid,rows):
        self.entry=entry;self.grid=grid
        self.coeff=CubicSpline(grid,[r["mean_polynomial"] for r in rows])
        self.correction=CubicSpline(grid,[r["mean_correction"] for r in rows])
        self.cov=covariance_interpolator(grid,np.array([r["covariance"] for r in rows]))
        self.risk=CubicSpline(grid,[r["variance"] for r in rows])
        self.center=(grid[0]+grid[-1])/2
        self.scale=(grid[-1]-grid[0])/2

    def mean(self,delta,b,corrected=True):
        coefficients=self.coeff(b)
        scale=np.asarray(delta)/.003
        result=sum(coefficients[...,i,:]*np.expand_dims(scale**i,-1) for i in range(5))
        return result+self.correction(b) if corrected else result

    def fit(self,data):
        # Global nuisance search, followed by multiple continuous starts.
        bg=np.linspace(self.grid[0],self.grid[-1],101)
        cov=self.cov(bg);prec=np.linalg.inv(cov);logdet=np.linalg.slogdet(cov)[1]
        mu=self.mean(0.,bg);a=self.coeff(bg)[:,1,:]/.003
        residual=data[:,None,:]-mu[None,:,:]
        pa=np.einsum("bij,bj->bi",prec,a)
        delta=np.clip(np.einsum("nbi,bi->nb",residual,pa)/np.einsum("bi,bi->b",a,pa),-.0035,.0035)
        residual=residual-delta[:,:,None]*a[None,:,:]
        loss=np.einsum("nbi,bij,nbj->nb",residual,prec,residual)+logdet
        fits=[]
        for i,y in enumerate(data):
            candidates=np.argsort(loss[i]);chosen=[]
            for index in candidates:
                if all(abs(bg[index]-bg[old])>.1875*self.scale for old in chosen):
                    chosen.append(index)
                if len(chosen)==4:break
            def objective(x):
                b=self.center+self.scale*x[1]
                c=self.cov(b);r=y-self.mean(x[0]*.001,b)
                return np.linalg.slogdet(c)[1]+r@np.linalg.solve(c,r)
            options=[minimize(objective,[delta[i,j]*1000,(bg[j]-self.center)/self.scale],method="L-BFGS-B",
                        bounds=[(-3.5,3.5),(-1.,1.)],
                        options=dict(ftol=1e-10,maxiter=80)) for j in chosen]
            best=min(options,key=lambda fit:fit.fun)
            fits.append([best.x[0]*.001,self.center+self.scale*best.x[1]])
        return np.array(fits)


def build_lookup(args):
    model=RamanD1(detuning_mhz=args.detuning)
    probability_fn=probability_d1
    if args.effective:
        from rb87_effective_compare import probability_effective
        probability_fn=probability_effective
    designs=make_designs(model)
    if args.refined_source:
        refined=json.loads((OUT/args.refined_source).read_text())
        assert refined["model"]==model.metadata()
        chosen=[r["entry"] for r in refined["entries"] if r["name"] in args.branches]
        assert len(chosen)==len(args.branches)
        if args.fixed_source:
            fixed=json.loads((OUT/args.fixed_source).read_text())
            designs[:2]=[r["entry"] for r in fixed["rows"] if r["detuning_mhz"]==args.detuning]
            assert len(designs[:2])==2
        designs=designs[:2]+chosen
    path=OUT/args.lookup
    if path.exists():
        saved=json.loads(path.read_text())
        assert saved["model"]==model.metadata()
        assert saved.get("effective",False)==args.effective
        assert saved["designs"]==designs
        assert np.array_equal(saved["grid"],np.linspace(-.8,.8,args.grid))
    else:
        saved=dict(model=model.metadata(),designs=designs,grid=np.linspace(-.8,.8,args.grid).tolist(),tables={},
                   refined_source=args.refined_source,fixed_source=args.fixed_source)
        saved["effective"]=args.effective
    for entry in designs:
        rows=saved["tables"].setdefault(entry["name"],[])
        for b in saved["grid"][len(rows):]:
            start=time.perf_counter();row=training_row(entry,b,model,probability_fn=probability_fn);rows.append(row)
            path.write_text(json.dumps(saved,indent=2))
            print(json.dumps(dict(name=entry["name"],b=b,variance=row["variance"],elapsed=time.perf_counter()-start)),flush=True)
    return saved


def experiment(args):
    saved=json.loads((OUT/args.lookup).read_text())
    model=RamanD1(detuning_mhz=saved["model"]["detuning_mhz"])
    probability_fn=probability_d1
    if saved.get("effective",False):
        from rb87_effective_compare import probability_effective
        probability_fn=probability_effective
    estimators=[Estimator(e,saved["grid"],saved["tables"][e["name"]]) for e in saved["designs"]]
    branches=estimators[2:];ncal=2
    grid=np.linspace(-.8,.8,161);dg=np.linspace(-.003,.003,21)
    prefix=branches[0];means=prefix.mean(0.,grid)[:,:ncal]
    cov=prefix.cov(grid)[:,:ncal,:ncal];jac=prefix.coeff(grid)[:,1,:ncal]/.003
    risk=np.array([e.risk(grid) for e in branches])
    central=abs(grid)<=.60001;fixedbranch=int(np.argmin(np.mean(risk[:,central],axis=1)))
    for e in branches[1:]:
        assert np.max(abs(e.mean(0.,grid)[:,:ncal]-means))<1e-8
    result=dict(model=model.metadata(),reps=args.reps,seed=args.seed,true_delta=args.delta,substeps=args.substeps,
        scope="OU trajectory and binomial data from full D1 or matched coherent qubit as marked; approximate Gaussian marginal likelihood. Frozen designs from the named lookup. No literature-optimality claim.",
        effective=saved.get("effective",False),paired_randomness=args.paired,
        covariance_interpolation={e.entry["name"]:e.cov.interpolation_method for e in estimators},
        lookup=args.lookup,branch_names=[e.entry["name"] for e in branches],
        fixed_branch=fixedbranch,resources=[dict(name=e.entry["name"],**optical_resources(e.entry["design"],e.entry["config"],model)) for e in estimators],rows=[])
    for b in args.points:
        rng=np.random.default_rng(args.seed+round((b+.8)*10000))
        d,c=prefix.entry["design"],prefix.entry["config"]
        noise,flags,phases=trajectories(d,c,b,args.reps,rng,args.delta,substeps=args.substeps)
        paired_noise={};uniform=None
        if args.paired:
            # A shared OU path on the union of physical pulse times couples
            # Monte Carlo arms without changing any arm's distribution.
            times=[];shapes=[]
            for e in estimators[:2]+[prefix]:
                dd,cc=e.entry["design"],e.entry["config"]
                ff=layout(dd["parameters"],dd["pattern"],cc)[0]
                centers=geometry(dd["parameters"],dd["family"],ff,cc,args.substeps)[0]
                times.append(centers.ravel());shapes.append(centers.shape)
            union=np.unique(np.concatenate(times));path=np.empty((args.reps,2,len(union)))
            std=c["sigma_intensity"]*np.sqrt(np.array([1+c["beam_correlation"],1-c["beam_correlation"]])/2)
            path[:,:,0]=rng.normal(size=(args.reps,2))*std
            for j in range(1,len(union)):
                rr=np.exp(-(union[j]-union[j-1])/c["correlation_time"])
                path[:,:,j]=rr*path[:,:,j-1]+np.sqrt(1-rr*rr)*rng.normal(size=(args.reps,2))*std
            for e,t,shape in zip(estimators[:2]+[prefix],times,shapes):
                paired_noise[e.entry["name"]]=path[:,:,np.searchsorted(union,t)].reshape(args.reps,2,*shape).transpose(0,2,1,3)
            noise=paired_noise[prefix.entry["name"]]
            uniform=rng.uniform(1e-12,1-1e-12,(args.reps,len(flags)))
        counts=atom_counts(d["parameters"],d["family"],flags,c)
        # The common prefix has no access to b except through physical data.
        # Evaluate only its two interrogations; preserving flags is necessary.
        short_config=dict(c,frequency_offsets=np.asarray(c["frequency_offsets"])[:ncal].tolist(),
            segment_phase_offsets=np.asarray(c["segment_phase_offsets"])[:ncal].tolist())
        pcal=probability_fn(args.delta,b,noise[:,:ncal],d["parameters"],d["family"],flags[:ncal],
                          phases[:ncal],short_config,args.substeps,model)
        ycal=(binom.ppf(uniform[:,:ncal],counts[:ncal],pcal) if args.paired else rng.binomial(counts[:ncal],pcal))/counts[:ncal]
        post=posterior(ycal,means,cov,jac,dg)
        choices=np.argmin(post@risk.T,axis=1)
        for name in ("fixed_hyper","fixed_shape","fixed_bank","adaptive"):
            start=time.perf_counter();fits=np.empty((args.reps,2))
            if name in ("adaptive","fixed_bank"):
                selected=choices if name=="adaptive" else np.full(args.reps,fixedbranch)
                for branch in np.unique(selected):
                    ids=np.flatnonzero(selected==branch);e=branches[branch]
                    p=probability_fn(args.delta,b,noise[ids],d["parameters"],d["family"],flags,phases,e.entry["config"],args.substeps,model)
                    measured=binom.ppf(uniform[ids,ncal:],counts[ncal:],p[:,ncal:]) if args.paired else rng.binomial(counts[ncal:],p[:,ncal:])
                    y=np.column_stack((ycal[ids],measured/counts[ncal:]))
                    fits[ids]=e.fit(y)
            else:
                e=next(x for x in estimators if x.entry["name"]==name)
                dd,cc=e.entry["design"],e.entry["config"]
                nn,ff,pp=trajectories(dd,cc,b,args.reps,rng,args.delta,substeps=args.substeps)
                if args.paired:nn=paired_noise[e.entry["name"]]
                count=atom_counts(dd["parameters"],dd["family"],ff,cc)
                p=probability_fn(args.delta,b,nn,dd["parameters"],dd["family"],ff,pp,cc,args.substeps,model)
                observed=binom.ppf(uniform,count,p) if args.paired else rng.binomial(count,p)
                fits=e.fit(observed/count)
            error=(fits[:,0]-args.delta)**2
            row=dict(b=b,name=name,mse=float(np.mean(error)),mse_se=float(np.std(error,ddof=1)/np.sqrt(args.reps)),
                     bias=float(np.mean(fits[:,0]-args.delta)),estimates=fits.tolist(),
                     branch_counts=np.bincount(choices,minlength=len(branches)).tolist() if name=="adaptive" else None,
                     elapsed=time.perf_counter()-start)
            result["rows"].append(row);(OUT/args.output).write_text(json.dumps(result,indent=2))
            print(json.dumps({key:value for key,value in row.items() if key!="estimates"}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["train","experiment"])
    p.add_argument("--lookup",default="rb87_d1_lookup.json")
    p.add_argument("--output",default="rb87_d1_adaptive_pilot.json")
    p.add_argument("--grid",type=int,default=21)
    p.add_argument("--detuning",type=float,default=1000.)
    p.add_argument("--points",type=float,nargs="+",default=[-.6,.6])
    p.add_argument("--reps",type=int,default=300)
    p.add_argument("--seed",type=int,default=705184)
    p.add_argument("--delta",type=float,default=.0005)
    p.add_argument("--substeps",type=int,default=1)
    p.add_argument("--refined-source")
    p.add_argument("--fixed-source")
    p.add_argument("--branches",nargs="+",default=["branch_0","branch_1","branch_7"])
    p.add_argument("--effective",action="store_true")
    p.add_argument("--paired",action="store_true")
    args=p.parse_args()
    if args.mode=="train":build_lookup(args)
    else:experiment(args)


if __name__=="__main__":main()
