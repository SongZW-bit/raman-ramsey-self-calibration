"""Independent single-pulse and full-sequence adiabatic-limit checks."""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from model import rotate
from two_beam import probability
from drift_study import layout
from multilevel_check import probability_ml

OUT = Path(__file__).resolve().parent / "results"


def main():
    single = []
    for detuning in (1e2, 1e3, 1e4, 1e5):
        errors = []
        for phase in (0., .7, -1.2, np.pi / 2):
            for duration in (.3, np.pi / 2, np.pi):
                for b in (0., -.6, .6):
                    a = np.sqrt(1 + b*b)
                    h = np.diag([.13, 0., detuning]).astype(complex)
                    h[0, 2] = h[2, 0] = np.sqrt(2*detuning*(a-b))/2
                    h[1, 2] = -np.sqrt(2*detuning*(a+b))*np.exp(1j*phase)/2
                    h[2, 1] = h[1, 2].conjugate()
                    # A coherent initial state tests phase signs invisible to a
                    # single population transfer from a basis state.
                    psi = np.array([1., 1j, 0.])/np.sqrt(2)
                    evolved = expm(-1j*h*duration) @ psi
                    bloch = rotate(np.array([0., 1., 0.]),
                                   np.array([np.cos(phase), np.sin(phase), .13+b]), duration)
                    errors.append(abs(abs(evolved[1])**2 - (1-bloch[2])/2))
        single.append(dict(detuning=detuning, max_error=float(max(errors))))
    fixed = json.loads((OUT / "two_beam_fixed_band_mc.json").read_text())
    sequences = []
    rng = np.random.default_rng(34825)
    for entry in fixed["designs"]:
        design, config = entry["design"], entry["config"]
        v = np.asarray(design["parameters"])
        for ratio in (False, True):
            cfg = dict(config, ratio_modulation=ratio,
                       segment_phase_offsets=rng.uniform(-.8,.8,(config["shots"],6)).tolist())
            if ratio:
                # Add actual nonzero offsets, independent of saved optimization.
                vtest = np.r_[v[:(12 if design["family"] == "shape" else 8)], .5, .2,-.1,-.25,.15]
            else:
                vtest = v
            flags, phases, _, _ = layout(vtest, design["pattern"], cfg)
            noise = rng.normal(0,.02,(5,len(flags),2,6))
            bs = np.linspace(-.6,.6,5)
            effective = probability(.0005,bs,noise,vtest,design["family"],flags,phases,cfg)
            for detuning in (1e2,1e3,1e4,1e5):
                rho = probability_ml(.0005,bs,noise,vtest,design["family"],flags,phases,cfg,
                                     detuning,return_density=True)
                error = float(np.max(abs(rho[...,1,1].real-effective)))
                sequences.append(dict(family=design["family"],ratio=ratio,detuning=detuning,
                    max_error=error, max_excited=float(rho[...,2,2].real.max()),
                    trace_error=float(np.max(abs(np.trace(rho,axis1=-2,axis2=-1)-1))),
                    min_eigenvalue=float(np.linalg.eigvalsh(rho).min())))
    assert single[-1]["max_error"] < 1e-4
    assert max(x["max_error"] for x in sequences if x["detuning"] == 1e5) < 3e-4
    assert max(x["trace_error"] for x in sequences) < 1e-9
    assert min(x["min_eigenvalue"] for x in sequences) > -1e-10
    result = dict(single_pulse=single,sequences=sequences,passed=True,
        scope="Closed Lambda mapping only; not a species-specific multilevel advantage test.")
    (OUT / "lambda_mapping_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
