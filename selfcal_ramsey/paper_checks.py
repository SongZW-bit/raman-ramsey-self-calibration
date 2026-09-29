"""Paper-specific square-pulse curve and independent full-sequence checks."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import root
from model import PHASES, probabilities

OUT = Path(__file__).resolve().parent/'results'
PAULI = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]], complex)


def hilbert_probability(x, v, shaped, dark, read_phase, gamma):
    delta, b, e = x
    intensity = v[6]*(1+e)
    a, c = v[7:9] if shaped else (0.,0.)
    rho = np.diag([1.,0.]).astype(complex)
    for pulse in range(2):
        phases = (0.,a,c) if pulse==0 else (-c,-a,0.)
        for phi in phases:
            phi += read_phase if pulse else 0.
            omega = [intensity*np.cos(phi), intensity*np.sin(phi),delta+b*intensity]
            u = expm(-.5j*np.einsum('i,ijk->jk',omega,PAULI)*v[pulse]/3)
            rho = u@rho@u.conj().T
        if pulse==0:
            u=expm(-.5j*delta*PAULI[2]*dark)
            rho=u@rho@u.conj().T
            rho[0,1]*=np.exp(-gamma*dark)
            rho[1,0]*=np.exp(-gamma*dark)
    return float(rho[1,1].real)


def main():
    OUT.mkdir(exist_ok=True)
    rows=[]
    # Fig. 2(b), PRApplied 9, 054034: equal pulses at r*pi/2.
    # This is a reconstruction of the published model, not digitized data.
    for r in (.8,1.,1.2):
        previous=0.
        for b in np.linspace(0.,.8,81):
            v=[r*np.pi/2,r*np.pi/2,1.,20.,.5,.8,1.]
            def err(z):
                p=probabilities([[z[0],b,0.]],v,correction=z[1])[0]
                return p[:,0]-p[:,1]
            fit=root(err,[0.,previous],options={'xtol':1e-10})
            assert np.linalg.norm(err(fit.x))<1e-10
            previous=float(fit.x[1])
            q=np.sqrt(1+b*b)
            analytic=2*np.arctan2(b*np.sin(q*r*np.pi/4),q*np.cos(q*r*np.pi/4))
            rows.append([r,b,fit.x[0],fit.x[1],analytic])
    with (OUT/'gabrs_fig2_model.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['pulse_area_factor','shift_over_rabi','lock_delta','numerical_phase','analytic_phase'])
        w.writerows(rows)
    rng=np.random.default_rng(8145)
    errors=[]
    for shaped in (False,True):
        for _ in range(12):
            x=rng.uniform([-.03,-.8,-.2],[.03,.8,.2])
            v=[*rng.uniform(.5,3.,2),.7,22.,.5,.8,.9]
            if shaped:v+=rng.uniform(-2.,2.,2).tolist()
            p=probabilities([x],v,shaped)[0]
            for j,dark in enumerate(v[2:4]):
                for k,phase in enumerate(PHASES):
                    errors.append(abs(p[j,k]-hilbert_probability(x,v,shaped,dark,phase,.02)))
    result=dict(source='https://doi.org/10.1103/PhysRevApplied.9.054034',
                scope='Fig. 2(b) pulse-area settings, positive shift interval [0,0.8]; model reconstruction, no pixel digitization.',
                max_phase_formula_error=float(max(abs(z[3]-z[4]) for z in rows)),
                max_lock_delta=float(max(abs(z[2]) for z in rows)),
                independent_sequence_cases=len(errors),
                max_sequence_probability_error=float(max(errors)))
    assert result['max_phase_formula_error']<1e-9 and max(errors)<1e-12
    (OUT/'paper_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
