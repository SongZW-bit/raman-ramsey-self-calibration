"""Paired finite-count intervals for the secular minor-helicity sensitivity test."""
import json
import numpy as np
from study import DATA


def run():
    source=json.loads((DATA/'impurity_mc.json').read_text())
    if len(source['rows'])!=6:raise RuntimeError('Impurity ensemble incomplete')
    rng=np.random.default_rng(263501);out={}
    for name in ('fixed_shape','fixed_square'):
        cells=[r for r in source['rows'] if r['name']==name]
        results=[]
        for j in (1,2):
            baseline=[];compared=[]
            for row in cells:
                pair=np.column_stack((row['scenarios'][0]['loss'],row['scenarios'][j]['loss']))
                baseline.append(pair[:,0].mean());compared.append(pair[:,1].mean())
            samples=np.zeros((6000,2))
            for row in cells:
                pair=np.column_stack((row['scenarios'][0]['loss'],row['scenarios'][j]['loss']))
                ix=rng.integers(0,len(pair),(6000,len(pair)))
                samples+=pair[ix].mean(axis=1)/len(cells)
            initial=float(np.mean(baseline));changed=float(np.mean(compared))
            results.append(dict(minor_helicity=cells[0]['scenarios'][j]['minor_helicity'],
                ideal_mse_hz2=initial*1e8,impurity_mse_hz2=changed*1e8,
                relative_change=changed/initial-1,
                relative_ci=np.quantile(samples[:,1]/samples[:,0]-1,[.025,.975]).tolist()))
        out[name]=results
    (DATA/'impurity_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))


if __name__=='__main__':run()
