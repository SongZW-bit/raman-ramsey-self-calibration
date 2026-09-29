"""Frozen primary-model evaluation, process-parallel and cell-resumable."""
import argparse,json,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.stats import binom
from study import DATA,existing,add_lookup,run_cell,paired_noise,select,layout,atom_counts,RamanD1,probability_d1

SHARDS=DATA/'primary_cells'

def primary_estimators():
    _,est=existing(False)
    add_lookup(est,'extended_lookup_d1.json');add_lookup(est,'fixed_lookup_d1.json')
    return est

def oracle_cell(task):
    _,idx,b,delta,reps,seed=task
    from conditional_selector import Selector
    est=primary_estimators();branches=[est[f'branch_{j}'] for j in range(8)]
    entry=branches[0].entry;d,c=entry['design'],entry['config'];rng=np.random.default_rng(seed)
    noise=paired_noise([entry],reps,rng,1,c)['branch_0'];unif=rng.uniform(1e-12,1-1e-12,(reps,12))
    flags,phases=layout(d['parameters'],d['pattern'],c)[:2];count=atom_counts(d['parameters'],d['family'],flags,c)
    model=RamanD1(decay_resolution='unresolved_excited');losses=[];yc=None
    for j,e in enumerate(branches):
        p=probability_d1(delta,b,noise,d['parameters'],d['family'],flags,phases,e.entry['config'],1,model)
        y=binom.ppf(unif,count,p)/count
        if yc is None:yc=y[:,:2].copy()
        else:assert np.max(abs(y[:,:2]-yc))<1e-12
        fit=e.fit(y);losses.append((fit[:,0]-delta)**2)
    return dict(b=b,delta=delta,reps=reps,seed=seed,losses=np.array(losses).T.tolist(),
        choices_local=select(branches,yc).tolist(),
        choices_mc=Selector(DATA/'primary_selector_mc.npz').select(yc).tolist())

def worker(task):
    kind,i,b,d,n,seed=task;path=SHARDS/f'{kind}_{i:03d}.json'
    if path.exists():return str(path)
    if kind=='oracle':out=oracle_cell(task)
    else:out=run_cell(primary_estimators(),b,d,n,seed,False,
        ['fixed_shape','fixed_refined','adaptive8','adaptive8_mc'],steps=1,keep_data=True,
        decay_resolution='unresolved_excited',selector_file=DATA/'primary_selector_mc.npz')
    path.write_text(json.dumps(out));return str(path)

def tasks():
    rng=np.random.default_rng(259251)
    bs=rng.uniform(-.6,.6,24);ds=rng.uniform(-.002,.002,24);seeds=rng.integers(1,2**31,24)
    result=[('holdout',i,float(b),float(d),96,int(s)) for i,(b,d,s) in enumerate(zip(bs,ds,seeds))]
    grid=[(float(b),float(d)) for b in [-.6,-.45,-.3,-.15,0,.15,.3,.45,.6] for d in [-.002,0,.002]]
    result += [('broad',i,b,d,96,261001+103*i) for i,(b,d) in enumerate(grid)]
    result += [('oracle',i,float(b),.0005,96,262001+103*i) for i,b in enumerate(np.linspace(-.6,.6,7))]
    return result

def collect():
    for kind in ['holdout','broad','oracle']:
        selected=[t for t in tasks() if t[0]==kind];files=[SHARDS/f'{kind}_{t[1]:03d}.json' for t in selected]
        rows=[]
        for f in files:
            if f.exists():
                x=json.loads(f.read_text());rows.extend([x] if kind=='oracle' else x)
        result=dict(decay_resolution='unresolved_excited',substeps=1,
                    planned_cells=len(selected),completed_cells=sum(f.exists() for f in files),
                    rows=rows,cells=[t[2:] for t in selected])
        (DATA/f'primary_{kind}.json').write_text(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('task',choices=['train','run','collect']);p.add_argument('--workers',type=int,default=5)
    a=p.parse_args();SHARDS.mkdir(exist_ok=True)
    if a.task=='train':
        from conditional_selector import train
        train(primary=True)
    elif a.task=='collect':collect()
    else:
        work=[t for t in tasks() if not (SHARDS/f'{t[0]}_{t[1]:03d}.json').exists()]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(worker,t):t for t in work}
            for f in as_completed(futures):
                print(json.dumps(dict(completed=f.result())),flush=True);collect()
        collect()

if __name__=='__main__':main()
