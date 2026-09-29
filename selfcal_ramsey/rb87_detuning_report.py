"""Publication-style diagnostics, explicitly labelled as local screening."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rb87_multilevel_pilot import OUT


def main():
    source=json.loads((OUT/"rb87_detuning_signed_scan.json").read_text())
    rows=source["rows"];detunings=sorted(set(r["detuning_mhz"] for r in rows))
    summaries=[]
    for detuning in detunings:
        subset=[r for r in rows if r["detuning_mhz"]==detuning]
        fixed={name:np.mean([r["variance"] for r in subset if r["name"]==name]) for name in ("fixed_hyper","fixed_shape")}
        oracle=np.mean([min(r["variance"] for r in subset if r["b"]==b and r["name"].startswith("branch")) for b in (-.6,.6)])
        summaries.append(dict(detuning_mhz=detuning,kappa=subset[0]["kappa"],**fixed,oracle=float(oracle),
            oracle_reduction_vs_better_fixed_family=float(1-oracle/min(fixed.values())),
            max_leakage=max(r["max_population_outside_sensor_pair"] for r in subset),
            peak_field_rss_mhz=.01*np.sqrt(max(r["resources"]["normalized_field_squared_peak"] for r in subset))))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":150,"savefig.dpi":240})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    x=np.array(detunings)
    for key,label,color,style in [("fixed_hyper","Fixed Hyper-Ramsey family","#546273","o-"),
        ("fixed_shape","Fixed phase-shaped","#b36e25","s-"),("oracle","Oracle branch diagnostic","#207c75","^--")]:
        axes[0].plot(x,[r[key]/1e-8 for r in summaries],style,color=color,label=label,markersize=4)
    axes[0].set(ylabel="Local variance (units of $10^{-8}$)",xlabel="Single-photon detuning (MHz)")
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].plot(x,[r["kappa"] for r in summaries],"o-",color="#715d8a")
    axes[1].axhline(1,color=".65",ls=":")
    axes[1].set(ylabel="Differential-shift susceptibility $\\kappa$",xlabel="Single-photon detuning (MHz)")
    axes[2].plot(x,[100*r["max_leakage"] for r in summaries],"o-",color="#a94f59")
    axes[2].set(ylabel="Max. population outside sensor pair (%)",xlabel="Single-photon detuning (MHz)")
    for ax in axes:
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("Full D1 manifold: detuning tradeoff (frozen-control screening)",fontsize=12)
    fig.savefig(OUT/"rb87_detuning_screening.png")
    fig.savefig(OUT/"rb87_detuning_screening.pdf")
    result=dict(scope="Two endpoint average of local variances. Oracle knows b; not an adaptive finite-data MSE result. All controls require reoptimization before a strong-baseline claim.",rows=summaries)
    (OUT/"rb87_detuning_summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
