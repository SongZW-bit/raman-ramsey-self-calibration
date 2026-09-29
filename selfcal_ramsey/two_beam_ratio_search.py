"""Compare ratio-coded controls under the same actual optical peak cap."""
from pathlib import Path
import argparse
import json
import numpy as np
from two_beam_identifiable import optimize
from two_beam import resources

OUT=Path(__file__).resolve().parent/'results'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_quadratic_joint.json')
    parser.add_argument('--output',default='two_beam_ratio_joint.json')
    parser.add_argument('--families',nargs='+',default=['hyper','shape'])
    parser.add_argument('--modes',nargs='+',default=['group','alternating'])
    parser.add_argument('--starts',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    source=json.loads((OUT/args.source).read_text())
    rows=json.loads((OUT/args.output).read_text())['results'] if args.resume and (OUT/args.output).exists() else []
    for mode in args.modes:
        config=dict(source['config'],ratio_modulation=True,ratio_alternating=mode=='alternating',
                    peak_cap=float(np.sqrt(1+source['config']['b_width']**2)))
        for kind in args.families:
            base=min((r for r in source['results'] if r['family']==kind),key=lambda r:r['variance'])
            for start in range(args.starts):
                if any(r['family']==kind and r['ratio_mode']==mode and r['start']==start for r in rows):continue
                rng=np.random.default_rng(5203+start)
                ratios=np.zeros(4) if start==0 else rng.uniform(-.3,.3,4)
                v=np.r_[base['parameters'],ratios]
                cost=resources(v,base['pattern'],config)
                v[4]*=min(1.,config['peak_cap']/cost['peak'])
                design=dict(base,parameters=v.tolist())
                result=optimize(design,config,cycles=2,maxiter=args.iterations,threshold=36.)
                result.update(ratio_mode=mode,start=start,config=config,readout='ratio-'+mode,
                              search_version='smooth_endpoint_peak_constraints')
                rows.append(result)
                print(json.dumps(dict(family=kind,mode=mode,start=start,
                    variance=result['variance'],ratios=result['parameters'][-4:],
                    resources=result['resources'])),flush=True)
                (OUT/args.output).write_text(json.dumps(dict(config=config,results=rows,
                    scope='Per-row config is authoritative; same optical peak, exposure, time, and atom caps.'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
