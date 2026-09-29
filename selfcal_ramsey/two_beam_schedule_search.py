"""Enumerate fixed-count measurement orderings and optionally refine controls.

All calibration data may enter the final offline estimate. Future calibration
is counted; these results are not a causal clock stability comparison.
"""
from pathlib import Path
from itertools import combinations
import argparse
import json
import time
import numpy as np
from drift_study import layout
from two_beam_identifiable import optimize, evaluate

OUT=Path(__file__).resolve().parent/'results'


def reorder(design,config,long_indices):
    old_flags,old_phases,_,_=layout(design['parameters'],design['pattern'],config)
    flags=np.zeros(len(old_flags),bool);flags[list(long_indices)]=True
    phases=np.empty(len(flags))
    for flag in (False,True):phases[flags==flag]=old_phases[old_flags==flag]
    return dict(long_flags=flags.tolist(),phases=phases.tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_joint_coded.json')
    parser.add_argument('--output',default='two_beam_schedule_scan.json')
    parser.add_argument('--top',type=int,default=3)
    parser.add_argument('--refine',action='store_true')
    parser.add_argument('--families',nargs='+',default=['hyper','shape'])
    args=parser.parse_args()
    source=json.loads((OUT/args.source).read_text());config=dict(source['config'],integer_atoms=False)
    results=[];scans=[]
    for kind in args.families:
        started=time.monotonic()
        original=min((r for r in source['results'] if r['family']==kind),key=lambda r:r['variance'])
        v=original['parameters'];flags=layout(v,original['pattern'],config)[0]
        candidates=[]
        for indices in combinations(range(len(flags)),sum(flags)):
            pattern=reorder(original,config,indices)
            variance=evaluate(v,kind,pattern,config,np.linspace(-.6,.6,13))
            candidates.append(dict(long_indices=list(indices),variance=variance))
        candidates.sort(key=lambda r:r['variance'])
        for row in candidates[:24]:
            pattern=reorder(original,config,row['long_indices'])
            row['dense_variance']=evaluate(v,kind,pattern,config,np.linspace(-.6,.6,81))
        shortlist=sorted(candidates[:24],key=lambda r:r['dense_variance'])[:args.top]
        scan=dict(family=kind,orderings=len(candidates),shortlist=shortlist,
                  original_long_indices=np.flatnonzero(flags).tolist(),
                  original_variance=evaluate(v,kind,original['pattern'],config,np.linspace(-.6,.6,81)),
                  original_finite_pulse_variance=original['variance'],
                  elapsed_seconds=time.monotonic()-started)
        scans.append(scan);print(json.dumps(scan),flush=True)
        for rank,row in enumerate(shortlist):
            design=dict(original,pattern=reorder(original,config,row['long_indices']))
            if args.refine:
                design=optimize(design,config,cycles=2,maxiter=100,threshold=source.get('threshold',36.))
            else:design['variance']=row['dense_variance']
            design['ordering_rank']=rank;design['source']=args.source
            results.append(design)
            (OUT/args.output).write_text(json.dumps(dict(config=config,results=results,scans=scans,
                refined=args.refine,scope='Fixed 12 calibration/4 sensing shots; offline estimate; no global continuous-control guarantee.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
