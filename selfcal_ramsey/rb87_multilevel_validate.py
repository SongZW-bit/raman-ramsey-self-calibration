"""Physical and independent propagation checks for the full D1 manifold."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from rb87_multilevel import RamanD1, N, G1, G2, D, atomic

OUT = Path(__file__).resolve().parent/"results"


def main():
    started = time.perf_counter()
    model = RamanD1()
    h = model.hamiltonian(.001,.4,.75,phase=.6,common=.03,difference=-.01)
    l = model.generator(h)
    tr = np.eye(N).reshape(-1,order="F")
    inactive = np.setdiff1d(np.arange(N*N),model.active)
    closure = float(np.max(abs(l[np.ix_(inactive,model.active)])))
    rho = model.propagate(model.initial,h,1.3)
    rho_full = model.propagate(model.initial,h,1.3,reduced=False)
    no_decay = RamanD1(emission=False)
    u = expm(-1j*h*1.3)
    rho_unitary = u @ model.initial @ u.conj().T
    rho_liouville = no_decay.propagate(model.initial,h,1.3)
    args=(.001,.4,.75,0.,0.,.03,-.01,1.3)
    gauge=model.phase_vector(.6)
    fast=gauge*(model.pulse_map(*args)@(gauge.conj()*model.initial.reshape(-1,order="F")[model.active]))
    checks = dict(parity_closure_error=closure,
        reduced_vs_full=float(np.max(abs(rho-rho_full))),
        coherent_vs_liouville=float(np.max(abs(rho_unitary-rho_liouville))),
        phase_gauge_error=float(np.max(abs(fast-rho.reshape(-1,order="F")[model.active]))),
        trace_preservation=float(np.max(abs(tr@l))),trace_error=float(abs(np.trace(rho)-1)),
        hermiticity=float(np.max(abs(rho-rho.conj().T))),min_eigenvalue=float(np.linalg.eigvalsh(rho).min()),
        branching_error=float(np.max(abs(sum(np.sum(d*d,axis=1) for d in D.values())-1))))
    mappings = []
    for b in (-.6,0.,.6):
        for phase in (0.,.7,-1.2):
            h = model.hamiltonian(.02,b,.8,phase)
            coupling = h[8:,:8]
            effective = h[:8,:8]-coupling.conj().T @ np.diag(1/model.bare[8:]) @ coupling
            mappings.append(dict(b=b,phase=phase,
                shift_error=float(abs((effective[G1,G1]-effective[G2,G2]).real-(.02+.8*b))),
                coupling_error=float(abs(2*effective[G1,G2]-.8*np.exp(-1j*phase)))))
    assert closure < 1e-10
    assert checks["reduced_vs_full"] < 1e-7
    assert checks["coherent_vs_liouville"] < 1e-7
    assert checks["phase_gauge_error"] < 1e-7
    assert checks["trace_error"] < 1e-7
    assert checks["min_eigenvalue"] > -1e-8
    assert max(x["shift_error"] for x in mappings)<1e-10
    assert max(x["coupling_error"] for x in mappings)<1e-10
    result = dict(model=model.metadata(),checks=checks,nominal_mapping=mappings,
        sample_final_populations=np.diag(rho).real.tolist(),passed=True,elapsed=time.perf_counter()-started,
        scope="Physical solver checks; no adaptive gain established by this validation.")
    (OUT/"rb87_multilevel_validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
