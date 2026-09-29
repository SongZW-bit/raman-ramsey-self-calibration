"""Scientific figure separating detuning, pulse and model effects."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rb87_multilevel_pilot import OUT


def main():
    comparison=json.loads((OUT/"rb87_effective_comparison.json").read_text())
    scan=json.loads((OUT/"rb87_physical_detuning_scan.json").read_text())
    square=json.loads((OUT/"rb87_square_reference.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"figure.dpi":150,"savefig.dpi":240})
    fig,axes=plt.subplots(1,3,figsize=(12.8,3.9),layout="constrained")
    x=np.linspace(60,1050,300);axes[0].plot(x,4*x/814.5-1,color="#247e78")
    axes[0].axhline(0,color=".7",lw=.8);axes[0].axvline(203.625,color="#b47532",ls="--",lw=1)
    axes[0].set(xlabel="Hamiltonian detuning (MHz)",ylabel="Signed shift / Raman coefficient",
                title="Interference control")
    axes[0].annotate("Leading-order cancellation",xy=(203.625,0),xytext=(330,.6),fontsize=8,
                     arrowprops=dict(arrowstyle="->",color=".4"))
    colors={"fixed_hyper":"#596979","fixed_shape":"#ae703b"}
    labels={"fixed_hyper":"Hyper-Ramsey family","fixed_shape":"Phase-shaped"}
    for name in colors:
        rows=[r for r in scan["rows"] if r["name"]==name]
        axes[1].plot([r["detuning_mhz"] for r in rows],[r["mean_variance"]/1e-8 for r in rows],
                     "o-",color=colors[name],label=labels[name],markersize=4)
        rows=[r for r in comparison["rows"] if r["name"]==name and r["detuning_mhz"]==200]
        for key,style,label in [("full_variance","-","D1"),("effective_variance","--","qubit")]:
            axes[2].plot([r["reference_b"] for r in rows],[r[key]/1e-8 for r in rows],style,
                color=colors[name],label=labels[name]+", "+label)
    axes[1].set(xlabel="Hamiltonian detuning (MHz)",ylabel="Local variance / $10^{-8}$",
                title="Frozen physical controls")
    squarevar=square["rows"][0]["mean_variance"]/1e-8
    axes[2].axhline(squarevar,color="#247e78",ls=":",label="Ramsey, D1 (3-point mean)")
    axes[2].set(xlabel="Reference light-shift parameter",ylabel="Local variance / $10^{-8}$",
                title="Matched models at 200 MHz")
    for ax in axes:ax.grid(axis="y",alpha=.15)
    axes[1].legend(frameon=False,fontsize=8)
    axes[2].legend(frameon=False,fontsize=7,loc="center right")
    fig.suptitle("D1 Raman sensing: mechanism and reference audit",fontsize=12)
    fig.savefig(OUT/"rb87_mechanism_comparison.png")
    fig.savefig(OUT/"rb87_mechanism_comparison.pdf")
    print("Saved mechanism comparison. Local variance only; no literature-optimality claim.")


if __name__=="__main__":main()
