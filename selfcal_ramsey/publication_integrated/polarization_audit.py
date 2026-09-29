"""Leading secular polarization-impurity susceptibility near 200 MHz."""
import json

from study import DATA, RamanD1
from rb87_joint_design import transport_b


def run():
    reference=RamanD1()
    model=RamanD1(detuning_mhz=200.)
    rows=[]
    for bref in (-.6,0.,.6):
        b=float(transport_b(reference,model,[bref])[0])
        e1,e2=model.fields(b,1.)
        nominal=model.ap*e1**2+model.am*e2**2
        for minor1,minor2 in [(0.,0.),(.005,.005),(.01,.01),(.05,.05),
                               (.01,0.),(0.,.01),(.02,.01),(.01,.02)]:
            ap=(1-minor1)*model.ap+minor1*model.am
            am=(1-minor2)*model.am+minor2*model.ap
            shift=ap*e1**2+am*e2**2
            rows.append(dict(reference_b=bref,physical_b=b,minor_intensity_1=minor1,
                             minor_intensity_2=minor2,ideal_shift_hz=nominal*1e4,
                             impurity_shift_hz=shift*1e4,shift_change_hz=(shift-nominal)*1e4,
                             resonant_coupling_fraction=((1-minor1)*(1-minor2))**.5))
    (DATA/"polarization_audit.json").write_text(json.dumps(dict(
        model="second-order secular intensity mixture; rapidly oscillating wrong-helicity Raman terms omitted",
        detuning_mhz=200.,rows=rows),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=="__main__":run()
