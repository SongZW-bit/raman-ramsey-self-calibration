"""Plot audited correlation scans and export selected MC comparisons."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'results'


def value(row,tau):
    return next(z['variance'] for z in row['actual'] if z['tau']==tau)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='drift_correlation_scan.json')
    parser.add_argument('--tau',type=float,default=1000.)
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    data=json.loads((OUT/args.source).read_text())
    taus=data['taus'];rows=data['results']
    modes=('quasistatic','shot','OU')
    records=[];selected=[]
    for tau in taus:
        record={'tau':tau}
        for mode in modes:
            eligible=[r for r in rows if r['mode']==mode and (mode!='OU' or r['design_tau']==tau)]
            best=min(eligible,key=lambda r:value(r,tau))
            record[mode]=value(best,tau)
            if tau==args.tau:
                selected.append(dict(best,variance=value(best,tau),label=mode))
        record['gain_vs_stronger_baseline']=1-record['OU']/min(record['shot'],record['quasistatic'])
        records.append(record)
    config=dict(data['config'],correlation_time=args.tau)
    (OUT/f'drift_selected{args.tag}.json').write_text(json.dumps(dict(config=config,results=selected),indent=2),encoding='utf-8')
    (OUT/f'drift_scan_summary{args.tag}.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.4,4),layout='constrained')
    for mode,color,label in zip(modes,('#777777','#C05A2B','#007C91'),
            ('Designed for quasistatic noise','Designed for shotwise noise','Designed for finite correlation')):
        ax.plot(taus,[r[mode]/1e-8 for r in records],'o-',color=color,label=label)
    ax.set(xscale='log',xlabel='Intensity correlation time (normalized)',
           ylabel='Worst-case local variance / 1e-8',title=f"Finite-pulse audit: {config['atoms_per_shot']:,} atoms per shot")
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
    fig.savefig(OUT/f'drift_correlation_scan{args.tag}.png',dpi=180)
    fig.savefig(OUT/f'drift_correlation_scan{args.tag}.pdf')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
