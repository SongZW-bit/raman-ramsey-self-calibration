"""Explore physical detuning, noise susceptibility, scattering and information."""
import json
import argparse
import time
import numpy as np
from rb87_multilevel import RamanD1,probability_d1,G1,G2
from rb87_multilevel_pilot import physical_design,statistics,optical_resources,OUT
from drift_study import layout
from two_beam_adaptive import branch_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--detunings",type=float,nargs="+",default=[100.,150.,250.,300.,500.,1000.])
    p.add_argument("--output",default="rb87_detuning_scan.json")
    args=p.parse_args()
    fixed=json.loads((OUT/"two_beam_fixed_band_mc.json").read_text())
    adaptive=json.loads((OUT/"two_beam_adaptive_phase_band.json").read_text())
    result=dict(scope="Exploratory local variances and oracle diagnostics; fixed/adaptive controls frozen after physical ratio conversion. No finite-data advantage claim.",rows=[])
    for detuning in args.detunings:
        model=RamanD1(detuning_mhz=detuning)
        kappa=abs(model.ap-model.am)*model.field_product
        d,c=physical_design(adaptive["design"],adaptive["config"],model)
        entries=[]
        for e in fixed["designs"]:
            fd,fc=physical_design(e["design"],e["config"],model)
            entries.append(("fixed_"+fd["family"],fd,fc))
        for i in (1,2,6,7):
            entries.append(("branch_"+str(i),d,branch_config(c,adaptive["offsets"][i],2)))
        for b in (-.6,.6):
            for name,design,config in entries:
                if (b<0 and name in ("branch_6","branch_7")) or (b>0 and name in ("branch_1","branch_2")):
                    continue
                start=time.perf_counter();stat=statistics(design,config,b,model)
                v=np.asarray(design["parameters"]);flags,phases,_,_=layout(v,design["pattern"],config)
                rho=probability_d1(0.,b,np.zeros((1,len(flags),2,6)),v,design["family"],flags,phases,config,
                                   model=model,return_density=True)
                pop=np.diagonal(rho[0],axis1=-2,axis2=-1).real
                row=dict(detuning_mhz=detuning,kappa=kappa,b=b,name=name,variance=stat["variance"],
                    common_variance=stat["common_variance"],difference_variance=stat["difference_variance"],
                    shot_variance=stat["shot_variance"],
                    max_population_outside_sensor_pair=float(np.max(1-pop[:,G1]-pop[:,G2])),
                    max_excited_population=float(np.max(pop[:,8:].sum(axis=1))),
                    resources=optical_resources(design,config,model),elapsed=time.perf_counter()-start)
                result["rows"].append(row)
                (OUT/args.output).write_text(json.dumps(result,indent=2))
                print(json.dumps(row),flush=True)
        model.pulse_map.cache_clear()


if __name__=="__main__":main()
