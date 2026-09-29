"""Exact three-level Lambda check for the effective Raman control study.

The existing controls are retained.  Only the effective Bloch propagation is
replaced by a finite-detuning |g1>-|e>-|g2> Hamiltonian.  This first check is
closed-system (Gamma=0), so it isolates adiabatic-elimination error from
spontaneous-emission loss.
"""
from pathlib import Path
import json
import numpy as np
from two_beam import geometry, noise_kernels, atom_counts, ratio_matrix
from drift_study import layout
from two_beam_adaptive import branch_config

OUT = Path(__file__).resolve().parent / "results"


def _unitary(h, duration):
    values, vectors = np.linalg.eigh(h)
    phase = np.exp(-1j * values * duration)[..., None, :]
    return np.einsum("...ik,...k,...jk->...ij", vectors, phase[..., 0, :], np.conj(vectors))


def probability_ml(delta, b, noise, v, kind, flags, phases, config,
                   detuning=50.0, substeps=1, return_density=False):
    noise = np.asarray(noise, float)
    ns, k, channels, total = noise.shape
    assert channels == 2 and total == 6 * substeps
    extra = 10 if kind in ("shape", "flex") else 6
    frequency = np.broadcast_to(v[extra] + np.asarray(config.get("frequency_offsets", 0.0)), (k,))
    short_phase = v[extra + 1]
    _, angles, durations = geometry(v, kind, flags, config, substeps)
    delta = np.broadcast_to(np.asarray(delta).reshape(-1, 1), (ns, k))
    b = np.broadcast_to(np.asarray(b).reshape(-1, 1), (ns, k))
    a = np.sqrt(1 + b * b)
    q = ratio_matrix(v, flags, config)
    bq = b[:, :, None] * np.cosh(q)[None, :, :] + a[:, :, None] * np.sinh(q)[None, :, :]
    aq = a[:, :, None] * np.cosh(q)[None, :, :] + b[:, :, None] * np.sinh(q)[None, :, :]
    # The g1 energy carries the two-photon detuning.  The beam assignment and
    # signs below make the eliminated Hamiltonian have h_z=delta+shift and
    # h_x+ih_y=Omega*exp(i*phase), matching the Bloch model.
    i_g1 = 1 + noise[:, :, 0, :] + noise[:, :, 1, :]
    i_g2 = 1 + noise[:, :, 0, :] - noise[:, :, 1, :]
    if detuning <= 0 or np.any(i_g1 < 0) or np.any(i_g2 < 0):
        raise ValueError("Positive detuning and nonnegative beam intensities required")
    pulse_b = np.repeat(bq, 3 * substeps, axis=2)
    pulse_a = np.repeat(aq, 3 * substeps, axis=2)
    # With |g1> carrying the two-photon detuning, these assignments give
    # -(a-b) I1/2 +(a+b) I2/2 = b(1+c)-a*d after elimination.
    amp1 = np.sqrt(2 * detuning * v[4] * (pulse_a - pulse_b) * i_g1)
    amp2 = np.sqrt(2 * detuning * v[4] * (pulse_a + pulse_b) * i_g2)
    segment_phases = np.asarray(config.get("segment_phase_offsets", np.zeros((k, 6))))
    if segment_phases.shape != (k, 6):
        raise ValueError("Expected six phase offsets per interrogation")
    rho = np.zeros((ns, k, 3, 3), complex)
    rho[..., 0, 0] = 1.0
    n = total // 2
    for s, angle in enumerate(angles):
        second = s >= n
        phase = angle + segment_phases[None, :, s // substeps]
        if second:
            phase = phase + phases[None, :] + v[5] + short_phase * (~flags)[None, :]
        h = np.zeros((ns, k, 3, 3), complex)
        laser_delta = delta - frequency[None, :]
        h[..., 0, 0] = laser_delta
        h[..., 2, 2] = detuning
        h[..., 0, 2] = amp1[..., s] / 2
        h[..., 2, 0] = h[..., 0, 2]
        h[..., 1, 2] = -amp2[..., s] * np.exp(1j * phase) / 2
        h[..., 2, 1] = np.conj(h[..., 1, 2])
        unitary = _unitary(h, durations[s])
        rho = np.einsum("...ij,...jk,...lk->...il", unitary, rho, np.conj(unitary))
        if s == n - 1:
            dark = np.where(flags, v[3], v[2])[None, :]
            z = np.ones((ns, k, 3), complex)
            z[..., 0] = np.exp(-1j * delta * dark)
            z[..., 2] = np.exp(-1j * detuning * dark)
            rho = np.einsum("...i,...ij,...j->...ij", z, rho, np.conj(z))
            if config.get("gamma", 0.0):
                dephase = np.exp(-config["gamma"] * dark)
                rho[..., 0, 1] *= dephase
                rho[..., 1, 0] *= dephase
                rho[..., :2, 2] *= dephase[..., None] ** .25
                rho[..., 2, :2] *= dephase[..., None] ** .25
    if return_density:
        return rho
    # The measured port is g2; excited-state population is an uncounted loss.
    return np.clip(np.real(rho[..., 1, 1]), 1e-10, 1 - 1e-10)


def local_statistics(design, config, b, detuning, branch_config_override=None):
    v = np.asarray(design["parameters"])
    kind = design["family"]
    flags, phases, _, _ = layout(v, design["pattern"], config)
    centers, _, durations = geometry(v, kind, flags, config, 1)
    k = len(flags)
    zero = np.zeros((1, k, 2, 6))
    h = 2e-5
    p = probability_ml(0., b, zero, v, kind, flags, phases, config, detuning)
    pd = probability_ml(h, b, zero, v, kind, flags, phases, config, detuning)
    md = probability_ml(-h, b, zero, v, kind, flags, phases, config, detuning)
    pb = probability_ml(0., b + h, zero, v, kind, flags, phases, config, detuning)
    mb = probability_ml(0., b - h, zero, v, kind, flags, phases, config, detuning)
    jac = np.column_stack(((pd[0] - md[0]) / (2 * h), (pb[0] - mb[0]) / (2 * h)))
    # Derivatives with respect to every finite-pulse noise sample.
    jnoise = np.zeros((k, 2 * k * 6))
    for channel in range(2):
        for shot in range(k):
            for segment in range(6):
                plus = zero.copy(); minus = zero.copy()
                plus[0, shot, channel, segment] = h
                minus[0, shot, channel, segment] = -h
                jnoise[shot, (channel * k + shot) * 6 + segment] = (
                    probability_ml(0., b, plus, v, kind, flags, phases, config, detuning)[0, shot]
                    - probability_ml(0., b, minus, v, kind, flags, phases, config, detuning)[0, shot]
                ) / (2 * h)
    times = centers.ravel()
    kc, kd = noise_kernels(times, config)
    kfull = np.zeros((2 * k * 6, 2 * k * 6))
    kfull[:k * 6, :k * 6] = kc
    kfull[k * 6:, k * 6:] = kd
    covariance = jnoise @ kfull @ jnoise.T
    counts = atom_counts(v, kind, flags, dict(config, integer_atoms=True))
    covariance += np.diag(p[0] * (1 - p[0]) / counts)
    precision = np.linalg.pinv(covariance, rcond=1e-12)
    info = jac.T @ precision @ jac
    variance = 1 / max(info[0, 0] - info[0, 1] ** 2 / max(info[1, 1], 1e-30), 1e-30)
    return dict(variance=float(variance), probabilities=p[0].tolist(),
                jacobian=jac.tolist(), covariance=covariance.tolist())


def main():
    fixed = json.loads((OUT / "two_beam_fixed_band_mc.json").read_text())
    adaptive = json.loads((OUT / "two_beam_adaptive_phase_band.json").read_text())
    shape = next(d for d in fixed["designs"] if d["design"]["family"] == "shape")
    design = shape["design"]
    config = shape["config"]
    adaptive_config = adaptive["config"]
    # Evaluate a representative branch bank from the adaptive calculation.
    bank = adaptive["offsets"]
    points = [-.6, -.4, .4, .6]
    rows = []
    for detuning in (100., 300., 1000., 3000.):
        for b in points:
            fixed_row = local_statistics(design, config, b, detuning)
            branches = []
            for control in bank:
                cfg = branch_config(adaptive_config, control, adaptive["calibration_shots"])
                branches.append(local_statistics(adaptive["design"], cfg, b, detuning))
            chosen = int(np.argmin([r["variance"] for r in branches]))
            rows.append(dict(detuning=detuning, b=b,
                             fixed_shape_variance=fixed_row["variance"],
                             adaptive_oracle_variance=branches[chosen]["variance"],
                             reduction=1 - branches[chosen]["variance"] / fixed_row["variance"],
                             selected_branch=chosen,
                             branch_variances=[r["variance"] for r in branches]))
    # Compare finite-detuning mean curves with the effective Lambda model.
    deviations = []
    from two_beam import probability as probability_effective
    for detuning in (100., 300., 1000., 3000.):
        for b in points:
            flags, phases, _, _ = layout(np.asarray(design["parameters"]), design["pattern"], config)
            noise = np.zeros((1, len(flags), 2, 6))
            p_ml = probability_ml(.0005, b, noise, np.asarray(design["parameters"]), "shape", flags, phases, config, detuning)
            p_eff = probability_effective(.0005, b, noise, np.asarray(design["parameters"]), "shape", flags, phases, config)
            deviations.append(dict(detuning=detuning, b=b,
                                   max_population_difference=float(np.max(abs(p_ml-p_eff))),
                                   rms_population_difference=float(np.sqrt(np.mean((p_ml-p_eff)**2)))))
    result = dict(model="closed three-level Lambda Hamiltonian", spontaneous_emission=False,
                  detuning_units="effective Raman Rabi frequency", points=points,
                  rows=rows, deviations=deviations,
                  scope="Local Gaussian variance with exact three-level propagation; adaptive branch is an oracle branch selection diagnostic, not a full adaptive Monte Carlo estimator.")
    (OUT / "multilevel_closed_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
