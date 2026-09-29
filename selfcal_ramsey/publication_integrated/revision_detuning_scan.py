"""Resolved-D1 local-risk comparison for frozen detuning-access designs."""
import json

import numpy as np

from study import DATA, OLD, RamanD1, statistics, probability_d1
from rb87_joint_design import transport_b


def run():
    reference=RamanD1()
    rows=[]
    for source in ("rb87_joint_phase_refined.json","rb87_square_reference.json",
                   "rb87_square_far_reference.json"):
        for old in json.loads((OLD/source).read_text())["rows"]:
            model=RamanD1(detuning_mhz=old["detuning_mhz"])
            points=transport_b(reference,model,[-.6,0.,.6])
            variance=[]
            for b in points:
                result=statistics(old["entry"]["design"],old["entry"]["config"],b,model,
                                  probability_fn=probability_d1)
                variance.append(float(result["variance"]))
            row=dict(name=old["name"],detuning_mhz=old["detuning_mhz"],
                     physical_b=np.asarray(points).tolist(),point_variance=variance,
                     mean_variance=float(np.mean(variance)),source=source)
            rows.append(row)
            (DATA/"resolved_detuning_scan.json").write_text(json.dumps(rows,indent=2))
            print(json.dumps(dict(name=row["name"],detuning_mhz=row["detuning_mhz"],
                                  mean_variance_hz2=row["mean_variance"]*1e8)),flush=True)


if __name__=="__main__":run()
