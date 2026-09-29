"""Independent physical and statistical checks for the two-beam extension."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from drift_study import layout,sequence
from drift_continuous import pulse_samples,statistics as old_statistics
from two_beam import probability,statistics,evaluate,geometry,atom_counts,ratio_matrix

OUT=Path(__file__).resolve().parent/'results'


def lambda_probability(delta,b,noise,v,kind,flags,phases,config,detuning):
    _,angles,durations=geometry(v,kind,flags,config,1)
    extra=10 if kind in ('shape','flex') else 6
    a=np.sqrt(1+b*b)
    ratios=ratio_matrix(v,flags,config)
    values=[]
    for j,flag in enumerate(flags):
        rho=np.zeros((3,3),complex);rho[0,0]=1.
        for s,angle in enumerate(angles):
            second=s>=3
            phase=angle+(phases[j]+v[5]+v[extra+1]*(not flag) if second else 0.)
            c,d=noise[0,j,:,s]
            q=ratios[j,int(second)]
            omega1=np.sqrt(2*detuning*v[4]*(a-b)*(1+c+d)*np.exp(-q))
            omega2=-np.sqrt(2*detuning*v[4]*(a+b)*(1+c-d)*np.exp(q))*np.exp(1j*phase)
            det=delta-v[extra]
            h=np.array([[det/2,0,omega1/2],[0,-det/2,omega2/2],
                        [omega1/2,np.conj(omega2)/2,detuning]],complex)
            u=expm(-1j*h*durations[s])
            rho=u@rho@u.conj().T
            if s==2:
                dark=v[3] if flag else v[2]
                u=np.diag(np.exp(-1j*np.array([delta/2,-delta/2,detuning])*dark))
                rho=u@rho@u.conj().T
        values.append(rho[1,1].real)
    return np.array(values)


def main():
    source=json.loads((OUT/'drift_selected.json').read_text())
    config=dict(source['config'],beam_correlation=1.,exposure_cap=160.,b_width=.6)
    row=source['results'][-1];v=row['parameters'];pattern=row['pattern'];kind=row['family']
    flags,phases,_,_=layout(v,pattern,config)
    rng=np.random.default_rng(219)
    x=np.array([[.0002,b,.01,-.02] for b in (-.6,0.,.6)])
    noises=np.zeros((3,len(flags),2,6));noises[:,:,0,:3]=x[:,2,None,None];noises[:,:,0,3:]=x[:,3,None,None]
    p=probability(x[:,0],x[:,1],noises,v,kind,flags,phases,config)
    old=sequence(x,v,kind,flags,phases,config['gamma'])
    common_error=float(np.max(abs(p-old)))
    assert common_error<1e-12
    shaped=v[:6]+[0.,0.,0.,0.]+v[6:]
    flex=shaped+[0.]*4
    assert np.max(abs(probability(x[:,0],x[:,1],noises,flex,'flex',flags,phases,config)-p))<1e-12
    oldvar=old_statistics(v,kind,pattern,.3,config,3)['variance']
    newvar=statistics(v,kind,pattern,.3,config,3,True)
    assert abs(newvar['variance']/oldvar-1)<1e-7
    assert np.max(abs(newvar['weights']@newvar['jacobian']-[1.,0.]))<1e-10
    allocated=list(v)+[.27]
    continuous=atom_counts(allocated,kind,flags,config)
    integers=atom_counts(allocated,kind,flags,dict(config,integer_atoms=True))
    assert integers.sum()==config['shots']*config['atoms_per_shot']
    assert np.max(abs(integers-continuous))<1
    allocated_stats=statistics(allocated,kind,pattern,.3,config,3,True)
    proj=np.sum(allocated_stats['weights']**2*allocated_stats['probabilities']*
                (1-allocated_stats['probabilities'])/continuous)
    assert abs(proj/allocated_stats['shot_variance']-1)<1e-12
    physical=[]
    cfg=dict(config,gamma=0.)
    noise=rng.normal(0,.015,(1,len(flags),2,6))
    for detuning in (100.,1000.,10000.,100000.):
        errs=[]
        for b in (-.6,0.,.6):
            predicted=probability(.0005,b,noise,v,kind,flags,phases,cfg)[0]
            exact=lambda_probability(.0005,b,noise,v,kind,flags,phases,cfg,detuning)
            errs.append(float(max(abs(predicted-exact))))
        physical.append(dict(single_photon_detuning=detuning,max_population_error=max(errs)))
    assert physical[-1]['max_population_error']<1e-4
    independent=dict(config,beam_correlation=0.)
    records=[]
    for n in (1,2,4,8):
        z=statistics(v,kind,pattern,.6,independent,n)
        records.append(dict(segments_per_pulse=3*n,**{k:z[k] for k in ('variance','shot_variance','common_variance','difference_variance')}))
    result=dict(common_mode_probability_error=common_error,common_mode_variance_relative_error=newvar['variance']/oldvar-1,
                lambda_comparison=physical,resolution=records,
                allocated_integer_total=int(integers.sum()),allocation_projection_check=True)
    (OUT/'two_beam_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
