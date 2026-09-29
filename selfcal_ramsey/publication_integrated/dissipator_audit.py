"""Resolved/nonsecular D1 comparison and integrated scattering accounting."""
import json

import numpy as np
from scipy.linalg import expm

from study import DATA, existing, layout, geometry, RamanD1
from rb87_multilevel import G1, G2, N
from two_beam import ratio_matrix
from rb87_joint_design import transport_b


def shot(model,entry,b,delta,shot_index):
    d,c=entry["design"],entry["config"]
    v=np.asarray(d["parameters"])
    flags,phases=layout(v,d["pattern"],c)[:2]
    _,angles,durations=geometry(v,d["family"],flags,c,1)
    q=ratio_matrix(v,flags,c)
    extra=10 if d["family"] in ("shape","flex") else 6
    frequency=v[extra]+np.asarray(c.get("frequency_offsets",np.zeros(len(flags))))[shot_index]
    offsets=np.asarray(c.get("segment_phase_offsets",np.zeros((len(flags),6))))
    ids=model.active
    excited=np.zeros(N*N)
    for j in range(8,16):excited[j+N*j]=5.7500/model.unit_mhz
    counter=excited[ids]
    vec=model.initial.reshape(-1,order="F")[ids].copy()
    count=0.
    for s,(angle,duration) in enumerate(zip(angles,durations)):
        second=int(s>=3)
        phase=angle+offsets[shot_index,s]
        if second:phase+=phases[shot_index]+v[5]+v[extra+1]*(not flags[shot_index])
        h=model.hamiltonian(delta,b,v[4],phase,frequency,q[shot_index,second])
        l=model.generator(h)[np.ix_(ids,ids)]
        augmented=np.zeros((len(ids)+1,len(ids)+1),complex)
        augmented[:-1,:-1]=l;augmented[-1,:-1]=counter
        z=expm(augmented*duration)@np.r_[vec,count]
        vec,count=z[:-1],z[-1].real
        if s==2:
            duration=v[3] if flags[shot_index] else v[2]
            h=np.diag(model.bare-delta*model.m/2)
            augmented[:-1,:-1]=model.generator(h,dark=True)[np.ix_(ids,ids)]
            z=expm(augmented*duration)@np.r_[vec,count]
            vec,count=z[:-1],z[-1].real
    whole=np.zeros(N*N,complex);whole[ids]=vec
    rho=whole.reshape((N,N),order="F")
    return dict(target=float(rho[G2,G2].real),initial=float(rho[G1,G1].real),
                spectator=float(sum(rho[j,j].real for j in range(N) if j not in (G1,G2))),
                emitted_photons=float(count),trace=float(np.trace(rho).real),
                min_eigenvalue=float(np.linalg.eigvalsh(rho).min()))


def run():
    entries,_=existing(False)
    source=json.loads((DATA/"resolved_200_lookup.json").read_text())
    e200=next(x["entry"] for x in source["entries"] if x["key"]=="200.0_fixed_shape")
    ref=RamanD1()
    rows=[]
    for detuning,entry in [(1000.,entries["fixed_shape"]),(1000.,entries["branch_0"]),(200.,e200)]:
        resolved=RamanD1(detuning_mhz=detuning,decay_resolution="excited_hyperfine")
        unresolved=RamanD1(detuning_mhz=detuning,decay_resolution="unresolved_excited")
        for bref in (-.6,0.,.6):
            b=float(transport_b(ref,resolved,[bref])[0])
            for index in (0,2):
                current=shot(resolved,entry,b,.0005,index)
                legacy=shot(unresolved,entry,b,.0005,index)
                rows.append(dict(detuning_mhz=detuning,entry=entry["name"],reference_b=bref,
                                 physical_b=b,shot=index+1,resolved=current,nonsecular=legacy))
                print(json.dumps(dict(detuning_mhz=detuning,entry=entry["name"],b=bref,
                                      shot=index+1,emitted=current["emitted_photons"],
                                      leakage=current["spectator"])),flush=True)
    (DATA/"dissipator_audit.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__":run()
