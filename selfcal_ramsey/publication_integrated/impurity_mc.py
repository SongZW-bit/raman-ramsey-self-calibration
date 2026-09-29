"""Finite-count secular-polarization sensitivity with frozen ideal estimators."""
import json
import numpy as np
from scipy.stats import binom
from study import DATA, OLD, Estimator, RamanD1, probability_d1, paired_noise, layout, atom_counts
from rb87_joint_design import transport_b


def run():
    sources=['rb87_joint_confirmation_lookup.json','rb87_square_lookup.json']
    entries=[]
    for source in sources:
        s=json.loads((OLD/source).read_text())
        entries.extend(e for e in s['entries'] if e['entry']['name'] in ('fixed_shape','fixed_square'))
    reference=RamanD1();nominal=RamanD1(detuning_mhz=200.)
    scenarios=[(0.,0.),(.01,.01),(.01,0.)]
    out=dict(model='secular minor-helicity Stark correction; wrong-helicity beat terms and their scattering omitted',
             decay_resolution='unresolved_excited',reps_per_cell=64,substeps=1,rows=[])
    path=DATA/'impurity_mc.json'
    if path.exists():out=json.loads(path.read_text())
    for entry in entries:
        e=entry['entry'];d,c=e['design'],e['config']
        estimator=Estimator(e,entry['grid'],entry['rows'])
        flags,phases=layout(d['parameters'],d['pattern'],c)[:2]
        count=atom_counts(d['parameters'],d['family'],flags,c)
        for bi,bref in enumerate((-.6,0.,.6)):
            if any(r['name']==e['name'] and r['reference_b']==bref for r in out['rows']):continue
            b=float(transport_b(reference,nominal,[bref])[0])
            rng=np.random.default_rng(263001+bi*103)
            noise=paired_noise([e],64,rng,1,c)[e['name']]
            unif=rng.uniform(1e-12,1-1e-12,(64,c['shots']))
            cell=dict(name=e['name'],reference_b=bref,physical_b=b,seed=263001+bi*103,scenarios=[])
            for impurity in scenarios:
                model=RamanD1(detuning_mhz=200.,minor_helicity=impurity)
                p=probability_d1(.0005,b,noise,d['parameters'],d['family'],flags,phases,c,1,model)
                y=binom.ppf(unif,count,p)/count
                fits=estimator.fit(y)
                loss=(fits[:,0]-.0005)**2
                cell['scenarios'].append(dict(minor_helicity=impurity,mse_hz2=float(loss.mean()*1e8),
                    loss=loss.tolist(),target_boundary_hits=int(np.sum(abs(fits[:,0])>.00349))))
            out['rows'].append(cell);path.write_text(json.dumps(out,indent=2))
            print(json.dumps({k:v for k,v in cell.items() if k!='scenarios'}|{'mse_hz2':[r['mse_hz2'] for r in cell['scenarios']]}),flush=True)


if __name__=='__main__':run()
