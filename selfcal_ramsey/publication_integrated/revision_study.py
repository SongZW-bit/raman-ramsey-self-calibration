"""Resolved-D1 follow-up; outputs never overwrite the original release."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from study import DATA, existing, clean, training_row, Estimator, run_cell


LOOKUP = DATA / "resolved_d1_lookup.json"
RESULT = DATA / "resolved_d1_holdout.json"
ENDPOINT = DATA / "resolved_d1_endpoints.json"


def designs():
    base, _ = existing(False)
    fixed = json.loads((DATA / "fixed_lookup_d1.json").read_text())["designs"][0]
    extended = json.loads((DATA / "extended_lookup_d1.json").read_text())["designs"]
    base.update({e["name"]: e for e in extended})
    return [clean(base["fixed_shape"]), clean(fixed)] + [clean(base[f"branch_{j}"]) for j in range(8)]


def train():
    entries = designs()
    if LOOKUP.exists():
        out = json.loads(LOOKUP.read_text())
        assert out["designs"] == entries and out["decay_resolution"] == "excited_hyperfine"
    else:
        out = dict(decay_resolution="excited_hyperfine", grid=np.linspace(-.8, .8, 21).tolist(),
                   designs=entries, tables={})
    from rb87_multilevel import RamanD1
    model = RamanD1(decay_resolution="excited_hyperfine")
    for entry in entries:
        rows = out["tables"].setdefault(entry["name"], [])
        for b in out["grid"][len(rows):]:
            start = time.perf_counter()
            rows.append(training_row(entry, b, model))
            LOOKUP.write_text(json.dumps(out, indent=2))
            print(json.dumps(dict(name=entry["name"], b=b, seconds=time.perf_counter()-start)), flush=True)


def estimators():
    s = json.loads(LOOKUP.read_text())
    assert all(len(s["tables"][e["name"]]) == len(s["grid"]) for e in s["designs"])
    return {e["name"]: Estimator(e, s["grid"], s["tables"][e["name"]]) for e in s["designs"]}


def holdout():
    est = estimators()
    if RESULT.exists():
        out = json.loads(RESULT.read_text())
        assert out["decay_resolution"] == "excited_hyperfine"
    else:
        rng = np.random.default_rng(259251)
        cells = [(float(b), float(d), 64, int(seed)) for b,d,seed in
                 zip(rng.uniform(-.6,.6,24), rng.uniform(-.002,.002,24),
                     rng.integers(1,2**31,24))]
        out = dict(decay_resolution="excited_hyperfine", sampling="independent uniform continuous holdout",
                   cells=cells, rows=[], completed_cells=0)
    for i, (b, delta, reps, seed) in enumerate(out["cells"]):
        if i < out["completed_cells"]:
            continue
        out["rows"].extend(run_cell(est,b,delta,reps,seed,False,
                                    ["fixed_shape","fixed_refined","adaptive8","adaptive8_mc"],
                                    steps=1,keep_data=True,decay_resolution="excited_hyperfine"))
        out["completed_cells"] = i+1
        RESULT.write_text(json.dumps(out, indent=2))


def endpoints():
    est = estimators()
    if ENDPOINT.exists():
        out = json.loads(ENDPOINT.read_text())
        assert out["decay_resolution"] == "excited_hyperfine"
    else:
        out = dict(decay_resolution="excited_hyperfine", cells=[[-.6,.0005,160,260001],
                   [.6,.0005,160,260002]], rows=[], completed_cells=0)
    for i,(b,delta,reps,seed) in enumerate(out["cells"]):
        if i < out["completed_cells"]:
            continue
        out["rows"].extend(run_cell(est,b,delta,reps,seed,False,
                                    ["fixed_shape","fixed_refined","adaptive8"],steps=1,
                                    keep_data=True,decay_resolution="excited_hyperfine"))
        out["completed_cells"] = i+1
        ENDPOINT.write_text(json.dumps(out,indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["train", "holdout", "endpoints"])
    task = p.parse_args().task
    {"train": train, "holdout": holdout, "endpoints": endpoints}[task]()
