"""Frozen-branch local variance decomposition for adaptive-control diagnosis."""
from pathlib import Path
import argparse
import json
import numpy as np
from two_beam_adaptive import branch_config
from two_beam_quadratic import statistics

OUT=Path(__file__).resolve().parent/'results'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='two_beam_adaptive_cal2_pilot.json')
    parser.add_argument('--output',default='two_beam_adaptive_mechanism.json')
    args=parser.parse_args()
    data=json.loads((OUT/args.source).read_text())
    design=data['design'];config=data['config'];rows=[]
    for observation in data['results']:
        if observation['mode']!='adaptive':continue
        b=observation['b'];branches=[]
        for offset in data['offsets']:
            cfg=branch_config(config,offset,data['calibration_shots'])
            row=statistics(design['parameters'],design['family'],design['pattern'],b,cfg,3,True)
            a=row['jacobian'][:,0]
            known_b=1/(a@np.linalg.solve(row['covariance'],a))
            components={key:row[key] for key in ('variance','shot_variance','common_variance','difference_variance','quadratic_variance')}
            components.update(known_b_local_variance=float(known_b),nuisance_penalty=float(row['variance']-known_b))
            assert abs(sum(components[k] for k in ('shot_variance','common_variance','difference_variance','quadratic_variance'))-row['variance'])<1e-15
            branches.append(components)
        proportions=np.array(observation['branch_counts'])/observation['reps']
        weighted={key:float(sum(p*r[key] for p,r in zip(proportions,branches))) for key in branches[0]}
        rows.append(dict(b=b,fixed=branches[data['fixed_branch']],selected_frozen_branch_average=weighted,
                         oracle_branch=int(np.argmin([r['variance'] for r in branches])),
                         observed_adaptive_mse=observation['mse']))
    result=dict(source=args.source,results=rows,
                scope='Diagnostic true-b local weights with frozen branch. Averaging by observed choice frequencies is not an attainable adaptive-risk bound or a decomposition of actual adaptive MSE.')
    (OUT/args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
