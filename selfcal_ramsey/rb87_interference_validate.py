"""Check closed-form D1 angular-factor formulas against atomic matrices."""
import json
import numpy as np
from rb87_multilevel import RamanD1
from rb87_joint_design import transport_b
from rb87_multilevel_pilot import OUT


def main():
    rows=[];reference=RamanD1()
    for detuning in [100.,150.,200.,250.,300.,500.,1000.]:
        model=RamanD1(detuning_mhz=detuning,field_gauss=0.)
        delta=detuning/model.unit_mhz;split=814.5/model.unit_mhz
        a=-(1/delta-5/(delta+split))/48
        c=(1/delta-1/(delta+split))/48
        relative=max(abs(model.ap/a-1),abs(model.am/(-a)-1),abs(model.cross/c-1))
        mapped=transport_b(reference,model,[-.6,0.,.6]);ratio_error=[]
        for bref,b in zip([-.6,0.,.6],mapped):
            e1,e2=reference.fields(bref,1.);f1,f2=model.fields(b,1.)
            ratio_error.append(abs(np.log(e1/e2)-np.log(f1/f2)))
        rows.append(dict(detuning_mhz=detuning,relative_coefficient_error=float(relative),
            kappa=float((model.ap-model.am)*model.field_product),
            analytic_kappa=4*detuning/814.5-1,maximum_log_ratio_error=float(max(ratio_error))))
    result=dict(rows=rows,passed=all(r["relative_coefficient_error"]<1e-11 and r["maximum_log_ratio_error"]<1e-11 for r in rows),
        scope="Second-order analytical coefficient and physical nuisance mapping validation; not a validation of the weak-coupling approximation at finite power.")
    (OUT/"rb87_interference_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
