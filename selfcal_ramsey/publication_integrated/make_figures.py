"""Rebuild publication figures and numerical tables from saved records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from study import ROOT, DATA, OLD, existing, optical_resources, RamanD1

FIG=ROOT/"figures";FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral"],"mathtext.fontset":"stix",
    "font.size":8.5,"axes.labelsize":9,"axes.titlesize":9,"legend.fontsize":7.5,
    "axes.spines.top":True,"axes.spines.right":True,"axes.linewidth":.6,
    "xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
    "xtick.major.width":.6,"ytick.major.width":.6,"lines.linewidth":1.0,
    "axes.grid":True,"grid.color":"#E4E7E9","grid.linewidth":.45,
    "pdf.fonttype":42,"ps.fonttype":42,
    "savefig.bbox":"tight","savefig.pad_inches":.04})
COL={"fixed_shape":"#252B30","fixed_refined":"#777E83","fixed_hyper":"#555D62","fixed_bank":"#252B30",
     "adaptive":"#245F9A","adaptive8":"#BF5044","adaptive8_mc":"#3B7A69",
     "frequency":"#BF5044","phase":"#3B7A69"}
LABEL={"fixed_shape":"Fixed shaped","fixed_refined":"Refined fixed","fixed_hyper":"Fixed HR family","fixed_bank":"Fixed bank",
       "adaptive":"Adaptive (3)","adaptive8":"Adaptive (8)","adaptive8_mc":"MC risk (8)",
       "frequency":"Frequency only","phase":"Phase only"}


def read(path):return json.loads(path.read_text())


def save(fig,name):
    fig.savefig(FIG/(name+".pdf"));fig.savefig(FIG/(name+".png"),dpi=180);plt.close(fig)


def paired(rows,reference="fixed_shape",adaptive="adaptive",seed=7712,boot=6000):
    rr=[r for r in rows if r["name"]==reference];aa=[r for r in rows if r["name"]==adaptive]
    assert len(rr)==len(aa)>0
    rng=np.random.default_rng(seed);fb=np.zeros(boot);ab=np.zeros(boot);fm=[];am=[]
    for f,a in zip(rr,aa):
        assert (f["b"],f["delta"],f["seed"])==(a["b"],a["delta"],a["seed"])
        fl=(np.array(f["estimates"])[:,0]-f["delta"])**2
        al=(np.array(a["estimates"])[:,0]-a["delta"])**2
        ids=rng.integers(0,len(fl),(boot,len(fl)))
        fb+=fl[ids].mean(axis=1)/len(rr);ab+=al[ids].mean(axis=1)/len(rr)
        fm.append(fl.mean());am.append(al.mean())
    f=float(np.mean(fm));a=float(np.mean(am));g=1-a/f
    return dict(reference=reference,adaptive=adaptive,fixed_mse=f,adaptive_mse=a,gain=g,
                ci=np.quantile(1-ab/fb,[.025,.975]).tolist(),strata=len(rr),records_per_arm=sum(r["reps"] for r in rr))


def summarize():
    out={}
    for name in ("broad","bank8","bank8_d1","d1","ablations"):
        p=DATA/(name+".json")
        if not p.exists():continue
        s=read(p);rows=s["rows"];out[name]={"cells":s["completed_cells"],"comparisons":[]}
        arms={r["name"] for r in rows}
        ref="fixed_bank" if name=="ablations" else ("fixed_refined" if name=="bank8_d1" else "fixed_shape")
        for arm in ("adaptive","adaptive8","frequency","phase"):
            if arm in arms:out[name]["comparisons"].append(paired(rows,ref,arm))
        if "fixed_refined" in arms and "adaptive" in arms:
            out[name]["comparisons"].append(paired(rows,"fixed_refined","adaptive"))
        if "fixed_refined" in arms and "adaptive8" in arms and ref!="fixed_refined":
            out[name]["comparisons"].append(paired(rows,"fixed_refined","adaptive8"))
        if name=="ablations":out[name]["comparisons"].append(paired(rows,"frequency","adaptive"))
        if name=="broad":
            out[name]["by_b"]=[dict(b=b,**paired([r for r in rows if r["b"]==b])) for b in sorted({r["b"] for r in rows})]
            out[name]["endpoint_comparison"]=paired([r for r in rows if abs(r["b"])==.6])
            out[name]["endpoint_refined"]=paired([r for r in rows if abs(r["b"])==.6],"fixed_refined")
        out[name]["tails"]={a:dict(target_bound_hits=sum(r["target_bound_hits"] for r in rows if r["name"]==a),
                  nuisance_bound_hits=sum(r["nuisance_bound_hits"] for r in rows if r["name"]==a),
                  max_abs_bias=max(abs(r["bias"]) for r in rows if r["name"]==a),
                  max_q99=max(r["abs_error_q99"] for r in rows if r["name"]==a)) for a in arms}
    for name in ("stress","d1_stress"):
        if not (DATA/(name+".json")).exists():continue
        rows=read(DATA/(name+".json"))["rows"];conditions=[]
        for r in rows:
            key=(r["noise"]["correlation_time"],r["noise"]["sigma_intensity"],r["noise"]["beam_correlation"],r["delay"])
            if key not in conditions:conditions.append(key)
        out[name]=[]
        for t,s,rho,l in conditions:
            subset=[r for r in rows if (r["noise"]["correlation_time"],r["noise"]["sigma_intensity"],r["noise"]["beam_correlation"],r["delay"])==(t,s,rho,l)]
            rr=paired(subset);rr.update(tau=t,sigma=s,rho=rho,delay=l)
            entries,_=existing(name=="stress")
            times={n:optical_resources(entries[n]["design"],entries[n]["config"],RamanD1())["elapsed"] for n in ("fixed_shape","branch_0")}
            rr["time_weighted_gain"]=1-(1-rr["gain"])*(times["branch_0"]+l)/times["fixed_shape"]
            out[name].append(rr)
    for name in ("convergence_effective","convergence_d1"):
        if not (DATA/(name+".json")).exists():continue
        rows=read(DATA/(name+".json"))["rows"];out[name]=[]
        for b in [-.6,.6]:
            for arm in ["fixed_shape","adaptive"]:
                rr=[r for r in rows if r["b"]==b and r["name"]==arm]
                if len(rr)!=2:continue
                x,y=[np.array(r["estimates"])[:,0] for r in rr]
                d=(y-.0005)**2-(x-.0005)**2
                rng=np.random.default_rng(77312);ids=rng.integers(0,len(d),(5000,len(d)))
                out[name].append(dict(b=b,arm=arm,relative_risk_change=float(d.mean()/np.mean((x-.0005)**2)),
                       difference_ci=np.quantile(d[ids].mean(axis=1),[.025,.975]).tolist(),
                       estimate_difference_rms=float(np.sqrt(np.mean((x-y)**2))),
                       branch_switches=None if arm!="adaptive" else int(np.sum(np.array(rr[0]["choices"])!=rr[1]["choices"]))))
    (DATA/"publication_summary.json").write_text(json.dumps(out,indent=2))
    return out


def panel(ax, text):
    ax.text(.035,.955,text,transform=ax.transAxes,ha='left',va='top',fontsize=9)


def risk_interval(rows, seed=61273):
    rng=np.random.default_rng(seed)
    boot=np.zeros(6000)
    for r in rows:
        loss=(np.asarray(r['estimates'])[:,0]-r['delta'])**2
        ids=rng.integers(0,len(loss),(6000,len(loss)))
        boot+=loss[ids].mean(axis=1)/len(rows)
    return np.quantile(boot,[.025,.975])


def primary_endpoint_gains():
    output=[]
    base=read(OLD/'rb87_refined_d1_mc200.json')
    for side,b in [('negative',-.6),('positive',.6)]:
        for effective in (True,False):
            raw=read(OLD/(f'rb87_effective_fine_{side}1600.json' if effective else f'rb87_resolution_{side}64.json'))
            rng=np.random.default_rng(27131); risks={};boots={}
            if effective:
                ix=rng.integers(0,1600,(6000,1600))
                for arm in ('fixed_shape','adaptive'):
                    r=next(r for r in raw['rows'] if r['name']==arm)
                    loss=(np.asarray(r['estimates'])[:,0]-raw['true_delta'])**2
                    risks[arm]=loss.mean();boots[arm]=loss[ix].mean(axis=1)
            else:
                ix=rng.integers(0,200,(6000,200));jx=rng.integers(0,64,(6000,64))
                for arm in ('fixed_shape','adaptive'):
                    r=next(r for r in base['rows'] if r['name']==arm and r['b']==b)
                    coarse=(np.asarray(r['estimates'])[:,0]-base['true_delta'])**2
                    losses={r['substeps']:(np.asarray(r['estimates'])[:,0]-raw['true_delta'])**2 for r in raw['rows'] if r['name']==arm}
                    diff=losses[3]-losses[1]
                    risks[arm]=coarse.mean()+diff.mean();boots[arm]=coarse[ix].mean(axis=1)+diff[jx].mean(axis=1)
            output.append(dict(b=b,effective=effective,reference='fixed_shape',adaptive='adaptive',
                               gain=float(1-risks['adaptive']/risks['fixed_shape']),
                               ci=np.quantile(1-boots['adaptive']/boots['fixed_shape'],[.025,.975]).tolist()))
    (DATA/'figure_endpoint_intervals.json').write_text(json.dumps(output,indent=2))
    return output


def schematic():
    from rb87_multilevel import atomic
    from drift_study import layout
    entries,_=existing(False);e=entries["branch_0"]
    v=e["design"]["parameters"];c=e["config"]
    fig,(ax,bx,cx)=plt.subplots(1,3,figsize=(7.05,2.67),
                                 gridspec_kw={"width_ratios":[1.12,.98,1.14]})
    ax.grid(False)
    for helicity,color,marker,label in [(1,COL["adaptive"],"o",r"$\sigma^+$"),
                                        (-1,COL["frequency"],"s",r"$\sigma^-$")]:
        yy,xx=np.nonzero(abs(atomic.D[helicity])>1e-12)
        strength=abs(atomic.D[helicity][yy,xx])**2
        ax.scatter(xx,yy,s=13+145*strength/np.max(abs(atomic.D[1])**2),
                   marker=marker,facecolors="white",edgecolors=color,linewidths=.85,
                   label=label,zorder=3)
    ax.axhline(2.5,color="#9BA3A8",lw=.65,ls=":")
    ax.axvline(2.5,color="#9BA3A8",lw=.65,ls=":")
    ax.set(xlim=(-.55,7.55),ylim=(-.55,7.55),aspect="equal",
           xlabel=r"Ground $m_F$",ylabel=r"Excited $m_{F'}$")
    ax.set_xticks(range(8),[str(m) for _,m in atomic.GROUND],fontsize=6.4)
    ax.set_yticks(range(8),[str(m) for _,m in atomic.EXCITED],fontsize=6.4)
    ax.legend(frameon=False,loc="upper left",ncol=2,fontsize=6.7,
              handletextpad=.25,columnspacing=.7,markerscale=.65)
    ax.text(.97,.97,"(a)",transform=ax.transAxes,ha="right",va="top",fontsize=9)
    flags=layout(v,e["design"]["pattern"],c)[0]
    shot=np.arange(1,len(flags)+1)
    dark=np.where(flags,v[3],v[2])*15.9155e-3
    bx.vlines(shot,0,dark,color="#C6CBCF",lw=.7,zorder=1)
    bx.axvline(2.5,color="#9BA3A8",lw=.65,ls=":")
    bx.scatter(shot[:2],dark[:2],s=24,marker="s",facecolor="white",edgecolor=COL["frequency"],
               lw=.9,label="Calibration",zorder=3)
    bx.scatter(shot[2:],dark[2:],s=20,marker="o",facecolor="white",edgecolor=COL["adaptive"],
               lw=.9,label="Measurement",zorder=3)
    bx.set(xlim=(.5,12.5),ylim=(-.14,2.5),xticks=[1,2,4,6,8,10,12],
           xlabel="Interrogation number",ylabel="Dark time (ms)")
    bx.legend(frameon=False,loc="center right",bbox_to_anchor=(.99,.66),
              fontsize=6.7,handletextpad=.25)
    panel(bx,"(b)")
    lookup=OLD/"rb87_refined_lookup.json"
    s=read(lookup);grid=np.asarray(s["grid"])
    for j in range(8):
        name=f"branch_{j}"
        source=s if name in s["tables"] else read(DATA/"extended_lookup_d1.json")
        y=np.array([r["variance"] for r in source["tables"][name]])*1e8
        cx.plot(grid,y,color="#A9B5BA",lw=.65,zorder=1)
    values=np.array([np.array([r["variance"] for r in (s if f"branch_{j}" in s["tables"] else read(DATA/"extended_lookup_d1.json"))["tables"][f"branch_{j}"]]) for j in range(8)])
    cx.plot(grid,values.min(axis=0)*1e8,color=COL["adaptive"],lw=1.5,label="Best local branch",zorder=3)
    cx.set(xlim=(-.8,.8),ylim=(1,500),yscale="log",xlabel=r"Reference light shift $b$",ylabel=r"Local variance (Hz$^2$)")
    panel(cx,"(c)")
    cx.legend(frameon=False,loc="lower right",fontsize=6.7)
    fig.subplots_adjust(left=.08,right=.99,bottom=.20,top=.97,wspace=.43)
    save(fig,"protocol")


def primary_and_domain(summary):
    fig=plt.figure(figsize=(7.05,3.7));gs=fig.add_gridspec(3,2,width_ratios=[.9,1.2],hspace=.06,wspace=.32)
    ax=fig.add_subplot(gs[:,0]);endpoint=primary_endpoint_gains()
    for effective,color,marker,ls,label in [(True,COL['adaptive'],'o','--','Effective two-level'),(False,COL['frequency'],'s','-','16-state D1')]:
        rr=[r for r in endpoint if r['effective']==effective];x=np.array([r['b'] for r in rr]);y=100*np.array([r['gain'] for r in rr]);ci=100*np.array([r['ci'] for r in rr])
        ax.errorbar(x,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=marker+ls,color=color,mfc='white',mew=.8,ms=4,capsize=2,elinewidth=.7,label=label)
    ax.axhline(0,color='.4',lw=.65,ls=':');ax.set(xlim=(-.8,.8),ylim=(-15,65),xlabel=r'Reference light shift $b$',ylabel='MSE reduction (%)');ax.set_xticks([-.6,0,.6]);panel(ax,'(a)')
    ax.legend(frameon=False,loc='lower center',fontsize=8)
    rows=read(DATA/'broad.json')['rows'];bs=sorted({r['b'] for r in rows})
    for j,delta in enumerate([-.0015,.0005,.002]):
        bx=fig.add_subplot(gs[j,1]);rr=[paired([r for r in rows if r['b']==b and r['delta']==delta]) for b in bs]
        y=100*np.array([r['gain'] for r in rr]);ci=100*np.array([r['ci'] for r in rr])
        bx.errorbar(bs,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt='o-',color=COL['adaptive'],mfc='white',mew=.7,ms=3.3,capsize=1.5,elinewidth=.6)
        bx.axhline(0,color='.4',lw=.6,ls=':');bx.set(xlim=(-.68,.68),ylim=(-80,80),yticks=[-50,0,50]);panel(bx,f'({chr(98+j)})  '+rf'$f={delta*1e4:g}$ Hz')
        bx.set_xticks([-.6,-.3,0,.3,.6])
        if j<2:bx.tick_params(labelbottom=False)
        else:bx.set_xlabel(r'Reference light shift $b$')
        if j==1:bx.set_ylabel('MSE reduction (%)')
    fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.98);save(fig,'risk_domain')


def controls_and_stress(summary):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.05,3.45),gridspec_kw={"width_ratios":[.83,1.35]})
    rows=read(DATA/"ablations.json")["rows"]
    for name,marker,ls in [('phase','^',':'),('frequency','s','--'),('adaptive','o','-')]:
        rr=[paired([r for r in rows if r['b']==b],'fixed_bank',name) for b in [-.6,.6]]
        y=100*np.array([r['gain'] for r in rr]);ci=100*np.array([r['ci'] for r in rr])
        ax.errorbar([-.6,.6],y,yerr=[y-ci[:,0],ci[:,1]-y],fmt=marker+ls,color=COL[name],mfc='white',mew=.8,ms=4,capsize=2,elinewidth=.65,label='Joint' if name=='adaptive' else LABEL[name])
    ax.axhline(0,color='.45',ls=':',lw=.6);ax.set(xlim=(-.8,.8),ylim=(-14,70),xlabel=r'Reference light shift $b$',ylabel='MSE reduction (%)');ax.set_xticks([-.6,0,.6]);ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(.02,.90),fontsize=8);panel(ax,'(a)')
    rows=summary["stress"];labs=[]
    for r in rows:
        if r["delay"]:lab=rf"Delay $L={r['delay']:g}$"
        elif r["tau"]!=300:lab=rf"$\tau={r['tau']:g}$"
        elif r["sigma"]!=.03:lab=rf"$\sigma_I={100*r['sigma']:g}\%$"
        elif r["rho"]:lab=rf"$\rho_I={r['rho']:g}$"
        else:lab="Nominal"
        labs.append(lab)
    for i,r in enumerate(rows):
        gain=r["gain"]*100;lo,hi=np.array(r["ci"])*100
        bx.plot([lo,hi],[i,i],color=COL["adaptive"],lw=.8);bx.plot(gain,i,"o",color=COL["adaptive"],mfc='white',mew=.8,ms=4)
        if r["delay"]:bx.plot(r["time_weighted_gain"]*100,i,"s",mfc="white",mec=COL['frequency'],ms=4)
    bx.axvline(0,color="#777777",lw=.7);bx.set_yticks(range(len(rows)),labs);bx.invert_yaxis()
    bx.set_xlabel("MSE reduction (%)");bx.set_ylim(9.8,-1.35);panel(bx,'(b)')
    bx.plot([],[],"s",mfc="white",mec=COL['frequency'],label=r"Risk $\times$ elapsed time");bx.legend(frameon=False,loc="lower right",fontsize=7)
    fig.subplots_adjust(wspace=.65,bottom=.16,top=.97);save(fig,"control_robustness")


def detuning():
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.05,2.8))
    delta=np.linspace(100,1000,401);k=4*delta/814.5-1
    ax.plot(delta,k,color=COL["adaptive"],lw=.95);ax.axhline(0,color="#888888",lw=.6)
    ax.axvline(203.625,color=COL['frequency'],ls="--",lw=.7)
    ax.annotate(r"$H/4=203.625$ MHz",(203.625,0),(320,.55),arrowprops=dict(arrowstyle="->",lw=.7),fontsize=9)
    ax.set_xlabel(r"Optical detuning $\Delta/2\pi$ (MHz)");ax.set_ylabel(r"Differential susceptibility $\kappa$")
    panel(ax,'(a)')
    sources=["rb87_joint_phase_refined.json","rb87_square_reference.json","rb87_square_far_reference.json"]
    data=[r for s in sources for r in read(OLD/s)["rows"]]
    for name,label,col,mark in [("fixed_hyper","HR family",COL["fixed_hyper"],"^"),("fixed_shape","Shaped",COL["fixed_shape"],"o"),("fixed_square","Ordinary Ramsey",COL["adaptive"],"s")]:
        rr=sorted([r for r in data if r["name"]==name],key=lambda r:r["detuning_mhz"])
        bx.plot([r["detuning_mhz"] for r in rr],[r["mean_variance"]*1e8 for r in rr],marker=mark,color=col,label=label,lw=.9,ms=4,mfc='white',mew=.8,ls={'fixed_hyper':':','fixed_shape':'--','fixed_square':'-'}[name])
    bx.set_yscale("log");bx.set_xlabel(r"$\Delta/2\pi$ (MHz)");bx.set_ylabel(r"Local variance (Hz$^2$)")
    panel(bx,'(b)');bx.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,.87))
    fig.subplots_adjust(wspace=.37,bottom=.2);save(fig,"detuning")


def mechanism():
    s=read(DATA/"theory.json");fig,(ax,bx)=plt.subplots(1,2,figsize=(7.05,2.65))
    for name,label,col,ls in [("branch_0","Branch 0",COL["fixed_bank"],':'),("branch_1","Branch 1",COL["adaptive"],'-'),("branch_7","Branch 7",COL["frequency"],'--')]:
        rr=[r for r in s["geometry"] if r["name"]==name]
        ax.plot([r["b"] for r in rr],[r["A"]*1e-7 for r in rr],label=label,color=col,lw=.9,ls=ls)
    ax.set_xlabel(r"Reference light shift $b$");ax.set_ylabel(r"Target information $A$ ($10^7$)");panel(ax,'(a)');ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=7,loc="upper center",ncol=1)
    t=np.linspace(0,4,200);bx.plot(t,np.exp(-2*t),color=COL["frequency"],label=r"$\exp(-2L/\tau)$")
    scale=s["ou"][0]["predicted_gain"]
    bx.errorbar([r["age_over_tau"] for r in s["ou"]],[r["observed_gain"]/scale for r in s["ou"]],
             yerr=[1.96*r["standard_error"]/scale for r in s["ou"]],fmt="o",ms=3.5,mfc='white',mew=.8,color=COL["adaptive"],label="Gaussian simulation")
    bx.set_xlabel(r"Calibration age $L/\tau$");bx.set_ylabel(r"$\Delta R(L)/\Delta R(0)$");panel(bx,'(b)');bx.legend(frameon=False,fontsize=7,loc='upper right');bx.set_ylim(-.04,1.15)
    fig.subplots_adjust(wspace=.4,bottom=.2);save(fig,"mechanism")


def bank_coverage():
    if not (DATA/"bank8.json").exists():return
    rows=read(DATA/"bank8.json")["rows"];fig,ax=plt.subplots(figsize=(3.4,2.5))
    for name,marker,ls in [("fixed_shape",'s',':'),("adaptive",'o','--'),("adaptive8",'^','-')]:
        rr=[r for r in rows if r["name"]==name]
        y=np.array([r['mse'] for r in rr])*1e8;ci=np.array([risk_interval([r]) for r in rr])*1e8
        ax.errorbar([r['b'] for r in rr],y,yerr=[y-ci[:,0],ci[:,1]-y],marker=marker,ms=3,lw=.8,ls=ls,color=COL[name],mfc='white',mew=.7,capsize=1.4,elinewidth=.5,label=LABEL[name])
    ax.set_xlabel(r"Reference light shift $b$");ax.set_ylabel(r"Frequency MSE (Hz$^2$)");ax.set_ylim(4.5,14);ax.legend(frameon=False,fontsize=8,ncol=1,loc='upper center')
    save(fig,"bank_coverage")


def primary_domain():
    holdout=DATA/"primary_holdout.json";oracle=DATA/"primary_oracle.json"
    if not holdout.exists() or not oracle.exists():return
    h=read(holdout);o=read(oracle)
    if h["completed_cells"]!=h["planned_cells"] or o["completed_cells"]!=o["planned_cells"]:return
    fig,axes=plt.subplots(2,2,figsize=(7.05,4.5),gridspec_kw={"height_ratios":[1,1.05]})
    ax,bx,cx,dx=axes.ravel();rows=h["rows"]
    keys=sorted({(r["b"],r["delta"],r["seed"]) for r in rows})
    points=[]
    for b,d,seed in keys:
        rr={r["name"]:r for r in rows if (r["b"],r["delta"],r["seed"])==(b,d,seed)}
        points.append((b,d,100*(1-rr["adaptive8"]["mse"]/rr["fixed_refined"]["mse"]),
                       rr["adaptive8"]["mse"]*1e8,rr["adaptive8_mc"]["mse"]*1e8))
    p=np.asarray(points)
    for mask,marker,color,label in [(p[:,1]<-.0005,"s",COL["frequency"],r"$f<-5$ Hz"),
                                    ((p[:,1]>=-.0005)&(p[:,1]<=.0005),"o",COL["adaptive"],r"$|f|\leq5$ Hz"),
                                    (p[:,1]>.0005,"^",COL["phase"],r"$f>5$ Hz")]:
        ax.scatter(p[mask,0],p[mask,2],marker=marker,s=22,facecolors="white",
                   edgecolors=color,linewidths=.8,label=label,zorder=3)
    ax.axhline(0,color="#72787C",lw=.7,ls=":")
    ax.set(xlim=(-.67,.67),xlabel=r"Reference light shift $b$",ylabel="MSE reduction (%)")
    ax.legend(frameon=False,fontsize=7,loc="lower left")
    panel(ax,"(a)")
    bx.plot([0,max(p[:,3:].max()*1.07,1)],[0,max(p[:,3:].max()*1.07,1)],color="#777E83",lw=.7,ls="--")
    bx.scatter(p[:,3],p[:,4],s=19,facecolors="white",edgecolors=COL["adaptive"],lw=.8)
    bx.set(xlim=(0,max(p[:,3:].max()*1.07,1)),ylim=(0,max(p[:,3:].max()*1.07,1)),
           xlabel="Local selector MSE (Hz$^2$)",ylabel="MC selector MSE (Hz$^2$)")
    panel(bx,"(b)")
    ordered=sorted(o["rows"],key=lambda r:r["b"])
    bs=np.array([r["b"] for r in ordered])
    oracle=np.array([np.asarray(r["losses"]).mean(axis=0).argmin() for r in ordered])
    for axis,field,color,letter in [(cx,"choices_local",COL["adaptive"],"(c)"),
                                    (dx,"choices_mc",COL["frequency"],"(d)")]:
        for r in ordered:
            probability=np.bincount(r[field],minlength=8)/len(r[field])
            used=np.flatnonzero(probability>0)
            axis.scatter(np.full(len(used),r["b"]),used,s=20+145*probability[used],
                         facecolor="white",edgecolor=color,lw=.85,zorder=3)
        axis.plot(bs,oracle,"x-",color="#252B30",lw=.65,ms=4,
                  label="Parameter oracle",zorder=4)
        axis.set(xlim=(-.69,.69),ylim=(-.55,7.55),yticks=range(8),
                 xlabel=r"True reference light shift $b$")
        panel(axis,letter)
    cx.set_ylabel("Selected branch")
    dx.legend(frameon=False,fontsize=7,loc="upper center")
    fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.96,wspace=.43,hspace=.48)
    save(fig,"domain_selection")


if __name__=="__main__":
    summary=summarize();schematic();primary_and_domain(summary);controls_and_stress(summary);detuning();mechanism();bank_coverage();primary_domain()
    print(json.dumps(summary,indent=2))
