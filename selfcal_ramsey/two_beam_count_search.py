"""Optimize interrogation count with a fixed total ensemble-atom budget.

These are restricted warm-started searches, not global design optima.
Every candidate retains its own authoritative configuration.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from two_beam import resources, atom_counts
from two_beam_identifiable import optimize
from two_beam_optimize import reshape

OUT=Path(__file__).resolve().parent/'results'


def resize_design(base, config, shots, nlong):
    config=dict(config,shots=shots,total_atoms=160000,atoms_per_shot=160000/shots,
                long_time_upper=300.)
    flags=np.zeros(shots,dtype=bool)
    flags[np.floor((np.arange(nlong)+.5)*shots/nlong).astype(int)]=True
    oldflags=np.asarray(base['pattern']['long_flags'],bool)
    oldphases=np.asarray(base['pattern']['phases'])
    phases=np.empty(shots)
    for flag in (False,True):
        ids=np.flatnonzero(flags==flag)
        phases[ids]=np.resize(oldphases[oldflags==flag],len(ids))
    v=np.array(base['parameters'])
    maximum=(config['time_cap']-shots*(config['overhead']+v[0]+v[1])-(shots-nlong)*v[2])/nlong
    if maximum<10.:return None
    v[3]=min(100.,maximum-1e-8)
    pattern=dict(long_flags=flags.tolist(),phases=phases.tolist())
    cost=resources(v,pattern,config)
    v[4]*=min(1.,config['exposure_cap']/cost['exposure'])
    counts=atom_counts(v,base['family'],flags,dict(config,integer_atoms=True))
    assert counts.sum()==160000 and min(counts)>0
    assert resources(v,pattern,config)['elapsed']<=config['time_cap']
    return dict(base,parameters=v.tolist(),pattern=pattern),config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_ratio_smooth.json')
    parser.add_argument('--output',default='two_beam_count_search.json')
    parser.add_argument('--counts',type=int,nargs='+',default=[8,10,12,14])
    parser.add_argument('--long-counts',type=int,nargs='+',default=[4])
    parser.add_argument('--families',nargs='+',default=['hyper','shape'])
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--cycles',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--embed-hyper',action='store_true')
    args=parser.parse_args()
    source=json.loads((OUT/args.source).read_text())
    rows=json.loads((OUT/args.output).read_text())['results'] if args.resume and (OUT/args.output).exists() else []
    for shots in args.counts:
        for nlong in args.long_counts:
            if not 2<=nlong<=shots-2:continue
            for kind in args.families:
                if any(r['family']==kind and r['config']['shots']==shots and r['nlong']==nlong for r in rows):continue
                candidates=[r for r in source['results'] if r['family']==('hyper' if args.embed_hyper else kind)]
                matched=[r for r in candidates if r.get('config',source['config'])['shots']==shots]
                base=min(matched or candidates,key=lambda r:r['variance'])
                if args.embed_hyper:
                    assert kind=='shape'
                    v=base['parameters']
                    base=dict(base,family='shape',parameters=reshape(v[:8],'hyper','shape')+v[8:])
                resized=resize_design(base,base.get('config',source['config']),shots,nlong)
                if resized is None:continue
                design,config=resized
                result=optimize(design,config,args.cycles,args.iterations,36.)
                result.update(config=config,nlong=nlong,readout=f'count-{shots}-long-{nlong}',
                              source=args.source)
                rows.append(result)
                (OUT/args.output).write_text(json.dumps(dict(config=config,results=rows,
                    scope='Per-row configs; fixed 160000 atoms, 2048 time, 160 exposure, original optical peak.'),indent=2),encoding='utf-8')
                print(json.dumps(dict(family=kind,shots=shots,nlong=nlong,variance=result['variance'],
                                      resources=result['resources'])),flush=True)


if __name__=='__main__':main()
