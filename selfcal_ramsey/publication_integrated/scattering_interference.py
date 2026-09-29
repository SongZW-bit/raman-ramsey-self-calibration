"""Low-saturation Kramers-Heisenberg path-interference diagnostic."""
import json
import numpy as np
from study import DATA,RamanD1
from rb87_multilevel import D,GROUND,EXCITED,G1


def run():
    rows=[]
    for detuning in (200.,1000.):
        model=RamanD1(detuning_mhz=detuning)
        for absorb_q in (-1,1):
            coherent=0.;separate=0.;channels=[]
            for f_index,(fg,mg) in enumerate(GROUND):
                for emit_q in (-1,0,1):
                    amplitudes=[]
                    for fe in (1,2):
                        amp=sum(D[emit_q][j,f_index]*D[absorb_q][j,G1]/
                                (model.bare[8+j]-model.bare[G1])
                                for j,(ff,mm) in enumerate(EXCITED) if ff==fe)
                        amplitudes.append(float(amp))
                    merged=float(sum(amplitudes)**2)
                    split=float(sum(a*a for a in amplitudes))
                    coherent+=merged;separate+=split
                    if merged or split:
                        channels.append(dict(final=[fg,mg],emitted_q=emit_q,
                                             amplitudes=amplitudes,coherent=merged,separate=split))
            rows.append(dict(detuning_mhz=detuning,absorbed_q=absorb_q,
                             coherent_sum=coherent,separate_sum=separate,
                             ratio=coherent/separate,channels=channels))
    (DATA/'scattering_interference.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='channels'} for r in rows],indent=2))


if __name__=='__main__':run()
