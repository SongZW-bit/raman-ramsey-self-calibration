"""Frozen-design transfer scans and cumulative differential-noise response."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from drift_study import layout
from two_beam import statistics,geometry

OUT=Path(__file__).resolve().parent/'results'


def main():
    candidates=[]
    for name in ('two_beam_two_allocated.json','two_beam_two-group_allocated.json'):
        source=json.loads((OUT/name).read_text())
        candidates.extend((r,source['config']) for r in source['results'])
    selected=[min((z for z in candidates if z[0]['family']==kind),key=lambda z:z[0]['variance'])
              for kind in ('hyper','shape')]
    rows=[];kernels=[]
    for row,config in selected:
        config=dict(config,integer_atoms=True)
        v=row['parameters'];kind=row['family'];pattern=row['pattern']
        result=statistics(v,kind,pattern,-.6,config,3,True)
        times=geometry(v,kind,layout(v,pattern,config)[0],config,3)[0].ravel()
        h=result['weights']@result['difference_jacobian']
        coefficient=config['sigma_intensity']**2*(1-config['beam_correlation'])*np.sum(np.cumsum(h)[:-1]**2*np.diff(times))
        assert abs(h.sum())<1e-8
        kernels.append(dict(family=kind,readout=row['readout'],times=times.tolist(),
                            cumulative=np.cumsum(h).tolist(),static_response=float(h.sum()),
                            slow_drift_coefficient=float(coefficient)))
        for rho in (-.5,0.,.5,1.):
            for tau in (30.,100.,300.,1000.,3000.):
                cfg=dict(config,beam_correlation=rho,correlation_time=tau)
                checks=[statistics(v,kind,pattern,b,cfg,2) for b in np.linspace(-.6,.6,25)]
                rows.append(dict(family=kind,rho=rho,tau=tau,
                                 worst_local_variance=max(z['variance'] for z in checks)))
    (OUT/'two_beam_mechanism.json').write_text(json.dumps(dict(
        note='Frozen controls and atom allocation; estimator weights adjust to assumed known noise spectrum. Not reoptimized pulse comparisons.',
        kernels=kernels,transfer_scan=rows),indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    colors={'hyper':'#23677b','shape':'#ad4256'}
    names={'hyper':'Composite baseline','shape':'Phase-shaped candidate'}
    for item in kernels:
        times=item['times'];values=item['cumulative']
        axes[0].step([times[0]-5]+times+[times[-1]+20],[0.]+values+[values[-1]],
                     where='post',color=colors[item['family']],label=names[item['family']])
    axes[0].axhline(0,color='.7',lw=.7);axes[0].set(xlabel='Time (normalized)',ylabel='Cumulative differential response',title='Pulse response and calibration timing')
    axes[0].legend(frameon=False,fontsize=8)
    for kind in ('hyper','shape'):
        subset=[r for r in rows if r['family']==kind and r['rho']==0]
        axes[1].loglog([r['tau'] for r in subset],[r['worst_local_variance'] for r in subset],
                       'o-',color=colors[kind],label=names[kind])
    axes[1].axvline(300,color='.7',ls=':',lw=1)
    axes[1].set(xlabel='Intensity correlation time (normalized)',ylabel='Worst local variance',title='Transfer with fixed pulse controls')
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(OUT/'two_beam_mechanism.png',dpi=200)
    fig.savefig(OUT/'two_beam_mechanism.pdf')
    print(json.dumps(dict(kernels=[{k:v for k,v in r.items() if k not in ('times','cumulative')} for r in kernels],transfer_scan=rows),indent=2))


if __name__=='__main__':main()
