"""Cross-model audit of every retrained control on an independent b grid."""
import json
import numpy as np
from rb87_multilevel import RamanD1
from rb87_multilevel_pilot import statistics,OUT
from rb87_effective_compare import probability_effective


def main():
    adaptive=json.loads((OUT/"rb87_adaptive_refined.json").read_text())
    fixed=json.loads((OUT/"rb87_fixed_band_refined.json").read_text())
    entries=[r["entry"] for r in fixed["rows"]]+[r["entry"] for r in adaptive["entries"]]
    model=RamanD1();result=dict(scope="Independent-grid local risk and oracle bounds, not actual adaptive MSE. Same frozen control in full D1 and polarizability-derived effective qubit.",rows=[])
    for b in [-.65,-.55,-.45,-.35,-.15,.15,.35,.45,.55,.65]:
        for e in entries:
            full=statistics(e["design"],e["config"],b,model)
            effective=statistics(e["design"],e["config"],b,model,probability_fn=probability_effective)
            row=dict(name=e["name"],b=b,full_variance=full["variance"],effective_variance=effective["variance"])
            result["rows"].append(row)
        (OUT/"rb87_refined_screen.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(b=b,best_fixed=min(r["full_variance"] for r in result["rows"][-len(entries):] if r["name"].startswith("fixed")),
             oracle=min(r["full_variance"] for r in result["rows"][-len(entries):] if r["name"].startswith("branch")))),flush=True)


if __name__=="__main__":main()
