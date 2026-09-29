"""Locked primary-model statistical summaries from independent cell shards."""
import json
import numpy as np
from study import DATA
from revision_summary import paired


def run():
    out={}
    for kind in ('holdout','broad'):
        path=DATA/f'primary_{kind}.json'
        if not path.exists():continue
        s=json.loads(path.read_text())
        if s['completed_cells']!=s['planned_cells']:continue
        comparisons=[('fixed_shape','adaptive8'),('fixed_refined','adaptive8'),
                     ('fixed_refined','adaptive8_mc'),('adaptive8','adaptive8_mc')]
        out[kind]=dict(cells=s['completed_cells'],records_per_cell=96,
            comparisons=[paired(s['rows'],a,b,kind=='holdout') for a,b in comparisons])
        if kind=='holdout':
            out[kind]['by_region']={}
            for label,lo,hi in [('negative',-.6,-.2),('interior',-.2,.2),('positive',.2,.6)]:
                rows=[r for r in s['rows'] if lo<=r['b']<=hi]
                out[kind]['by_region'][label]=paired(rows,'fixed_refined','adaptive8',True)
        out[kind]['selection_frequencies']={arm:np.bincount(
            np.concatenate([r['choices'] for r in s['rows'] if r['name']==arm]),minlength=8).tolist()
            for arm in ('adaptive8','adaptive8_mc')}
    path=DATA/'primary_oracle.json'
    if path.exists():
        s=json.loads(path.read_text())
        if s['completed_cells']==s['planned_cells']:
            out['oracle']=[]
            for r in s['rows']:
                losses=np.asarray(r['losses']);best=int(np.argmin(losses.mean(axis=0)))
                local=np.asarray(r['choices_local']);mc=np.asarray(r['choices_mc'])
                out['oracle'].append(dict(b=r['b'],best_parameter_branch=best,
                    branch_risk_hz2=(losses.mean(axis=0)*1e8).tolist(),
                    local_choices=np.bincount(local,minlength=8).tolist(),
                    mc_choices=np.bincount(mc,minlength=8).tolist(),
                    local_misselection=float(np.mean(local!=best)),
                    mc_misselection=float(np.mean(mc!=best)),
                    parameter_oracle_hz2=float(losses[:,best].mean()*1e8),
                    realization_oracle_hz2=float(losses.min(axis=1).mean()*1e8),
                    local_hz2=float(losses[np.arange(len(losses)),local].mean()*1e8),
                    mc_hz2=float(losses[np.arange(len(losses)),mc].mean()*1e8)))
    for name in ('resolved_200_mc.json','resolved_200_square_mc.json'):
        p=DATA/name
        if not p.exists():continue
        s=json.loads(p.read_text())
        if len(s['rows'])!=3*(2 if name=='resolved_200_mc.json' else 1):continue
        out[name]={}
        for key in sorted({r['key'] for r in s['rows']}):
            rows=[r for r in s['rows'] if r['key']==key]
            rng=np.random.default_rng(259290)
            boot=np.zeros(6000)
            for row in rows:
                loss=(np.asarray(row['estimates'])[:,0]-s['true_delta'])**2
                ids=rng.integers(0,len(loss),(6000,len(loss)))
                boot+=loss[ids].mean(axis=1)/len(rows)
            out[name][key]=dict(mse_hz2=float(np.mean([r['mse'] for r in rows])*1e8),
                mse_ci_hz2=(np.quantile(boot,[.025,.975])*1e8).tolist(),
                boundary_fraction=float(np.mean([r['boundary_fraction'] for r in rows])),
                records=sum(len(r['estimates']) for r in rows))
    (DATA/'primary_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':run()
