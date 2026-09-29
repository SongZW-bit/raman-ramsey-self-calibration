"""Summarize ratio modulation and count/allocation tests without mixing metrics."""
from pathlib import Path
import json
import numpy as np

OUT=Path(__file__).resolve().parent/'results'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def summary(source,family):
    data=read(source)
    rows=[r for r in data['results'] if r['family']==family]
    assert sorted(r['b'] for r in rows)==[-.6,-.3,0.,.3,.6], 'Wait for the complete five-point run'
    worst=max(rows,key=lambda r:r['delta_mse'])
    chosen=next(d for d in data['designs'] if d['design']['family']==family)
    return dict(source=source,family=family,points=len(rows),reps=worst['reps'],
                maximum_sampled_mse=worst['delta_mse'],worst_b=worst['b'],
                standard_error_at_worst=worst['mse_standard_error'],
                resources=chosen['design']['resources'],
                maximum_sampled_catastrophic_fraction=max(r['nuisance_catastrophic_fraction'] for r in rows))


def comparison(left,right):
    a=left['maximum_sampled_mse'];b=right['maximum_sampled_mse']
    return dict(reference=left['source']+':'+left['family'],candidate=right['source']+':'+right['family'],
                mse_reduction=1-b/a,rmse_reduction=1-np.sqrt(b/a))


def main():
    rows=[summary('two_beam_ratio_mc.json',kind) for kind in ('hyper','shape')]
    rows += [summary('two_beam_count_'+kind+'_mc.json',kind) for kind in ('hyper','shape')]
    comparisons=[comparison(rows[0],rows[1]),comparison(rows[2],rows[3]),
                 comparison(rows[0],rows[2]),comparison(rows[1],rows[3])]
    ratio=read('two_beam_ratio_mc.json')['results']
    pointwise=[]
    for hyper in (r for r in ratio if r['family']=='hyper'):
        shape=next(r for r in ratio if r['family']=='shape' and r['b']==hyper['b'])
        pointwise.append(dict(b=hyper['b'],hyper_mse=hyper['delta_mse'],shape_mse=shape['delta_mse'],
                              mse_reduction=1-shape['delta_mse']/hyper['delta_mse']))
    result=dict(results=rows,comparisons=comparisons,ratio_pointwise=pointwise,
                scope='Five-point Monte Carlo maxima, not continuous worst-case bounds. Count design selected by surrogate only. No literature-wide optimum claim.')
    (OUT/'ratio_count_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
