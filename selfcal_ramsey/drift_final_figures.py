"""Show the strongest checked pulse baseline alongside the square designs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drift_continuous import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    selected=json.loads((OUT/'drift_selected.json').read_text())
    hyper=json.loads((OUT/'drift_hyper_tau1000.json').read_text())
    mc=json.loads((OUT/'drift_selected_mc.json').read_text())['results']
    mc+=json.loads((OUT/'drift_hyper_tau1000_mc.json').read_text())['results']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.4,4),layout='constrained')
    summary=[]
    designs=[(r,selected['config']) for r in selected['results']]+[(hyper['results'][0],hyper['config'])]
    labels=['Quasistatic design','Shotwise-noise design','Finite-correlation design','Hyper-Ramsey-type baseline']
    colors=['#777777','#C05A2B','#007C91','#6B54A3']
    for (row,config),label,color in zip(designs,labels,colors):
        bs=np.linspace(-.6,.6,121)
        values=np.array([statistics(row['parameters'],row['family'],row['pattern'],b,config,3)['variance'] for b in bs])
        axes[0].plot(bs,values/1e-8,label=label,color=color)
        samples=[r for r in mc if r['design_label']==row['label']]
        axes[1].errorbar([r['b'] for r in samples],[r['delta_mse']/1e-8 for r in samples],
                         yerr=[1.96*r['mse_standard_error']/1e-8 for r in samples],
                         color=color,marker='o',markersize=3,linewidth=1.2,capsize=2,label=label)
        worst=max(samples,key=lambda z:z['delta_mse'])
        summary.append(dict(label=row['label'],local_variance_max=float(values.max()),
             sampled_worst_mse=worst['delta_mse'],sampled_worst_b=worst['b'],
             worst_point_mc_se=worst['mse_standard_error'],resources=row['resources']))
    axes[0].set(title='Local linear-estimator prediction',ylabel='Frequency variance / 1e-8')
    axes[1].set(title='Nonlinear trajectories and binomial readout',ylabel='Frequency MSE / 1e-8')
    for ax in axes:
        ax.set(xlabel='Unknown light-shift coefficient b')
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(OUT/'drift_strong_baseline.png',dpi=180)
    fig.savefig(OUT/'drift_strong_baseline.pdf')
    (OUT/'drift_final_summary.json').write_text(json.dumps(dict(correlation_time=1000.,
        mc_trials_per_point=5000,true_delta=.0005,
        caveat='MSE maxima are over nine tested b values, not continuous worst-case guarantees. Error bars are pointwise 1.96 MC standard errors.',
        results=summary),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
