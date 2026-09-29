"""Vectorized finite-pulse Bloch dynamics and counted self-calibration data."""
from dataclasses import dataclass
import numpy as np

PHASES = np.array([np.pi/2, -np.pi/2, np.pi/4, -3*np.pi/4])

@dataclass(frozen=True)
class Budgets:
    atoms: float = 100000.0
    elapsed: float = 2500000.0
    exposure: float = 400000.0
    overhead: float = 2.0
    gamma: float = .02


def rotate(r, omega, duration):
    norm = np.linalg.norm(omega, axis=-1)
    axis = omega / np.maximum(norm[..., None], 1e-30)
    angle = norm * duration
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    return r*c + np.cross(axis, r)*s + axis*np.sum(axis*r, axis=-1)[..., None]*(1-c)


def decode(v, shaped=False):
    # Both pulse families have identical amplitude and exposure constraints.
    # Shaped pulses contain the square baseline when both interior phases vanish.
    t1, t2, short, long, fraction_long, fraction_quad, power = v[:7]
    a, b = v[7:9] if shaped else (0., 0.)
    return t1, t2, short, long, fraction_long, fraction_quad, power, a, b


def probabilities(x, v, shaped=False, gamma=.02, correction=0.):
    x = np.atleast_2d(x)
    t1, t2, ts, tl, _, _, power, a, b = decode(v, shaped)
    # Shape is (parameter samples, dark settings, phase settings, Bloch axes).
    dims = (len(x), 2, 4)
    delta = np.broadcast_to(x[:, 0, None, None], dims)
    intensity = np.broadcast_to(power*(1+x[:, 2, None, None]), dims)
    z = delta + x[:, 1, None, None]*intensity
    r = np.zeros(dims+(3,)); r[..., 2] = 1.
    offset_index = 9 if shaped else 7
    offset = v[offset_index] if len(v) > offset_index else 0.
    read_phases = PHASES[None, None, :] + correction + offset
    for pulse, duration in enumerate((t1, t2)):
        phases = (0., a, b) if pulse == 0 else (-b, -a, 0.)
        for phase in phases:
            phi = phase + (read_phases if pulse else 0.)
            h = np.stack([intensity*np.cos(phi), intensity*np.sin(phi), z], axis=-1)
            r = rotate(r, h, duration/3.)
        if pulse == 0:
            dark = np.array([ts, tl])[None, :, None]
            h = np.zeros(dims+(3,)); h[..., 2] = delta
            r = rotate(r, h, dark)
            r[..., :2] *= np.exp(-gamma*dark)[..., None]
    return np.clip((1-r[..., 2])/2., 1e-14, 1-1e-14)


def weights(v):
    fl, fq = v[4:6]
    return np.outer([1-fl, fl], [fq/2, fq/2, (1-fq)/2, (1-fq)/2])


def resources(v, budgets=Budgets()):
    t1, t2, ts, tl, fl, _, power = v[:7]
    time = budgets.overhead+t1+t2+(1-fl)*ts+fl*tl
    exposure = power*(t1+t2)
    n = min(budgets.atoms, budgets.elapsed/time, budgets.exposure/exposure)
    return dict(atoms=n, time_per_atom=time, light_per_atom=exposure,
                elapsed=n*time, exposure=n*exposure, peak_power=power,
                short_atoms=n*(1-fl), long_atoms=n*fl)


def fisher(x, v, shaped=False, budgets=Budgets(), step=2e-5):
    x = np.atleast_2d(x)
    p = probabilities(x, v, shaped, budgets.gamma)
    d = []
    for j in range(3):
        shift = np.eye(3)[j]*step
        d.append((probabilities(x+shift, v, shaped, budgets.gamma)
                  - probabilities(x-shift, v, shaped, budgets.gamma))/(2*step))
    jac = np.stack(d, axis=-1)
    factor = weights(v)[None, :, :]/(p*(1-p))
    f = np.einsum('nsqi,nsqj,nsq->nij', jac, jac, factor)
    return f * resources(v, budgets)['atoms']


def effective_info(f):
    nuisance = f[:, 1:, 1:]
    cross = f[:, 0, 1:]
    # Projection, rather than regularization, handles locally redundant nuisance
    # directions. Finite-data checks are still needed at nonregular points.
    removed = np.einsum('ni,nij,nj->n', cross, np.linalg.pinv(nuisance, rcond=1e-10), cross)
    return np.maximum(0., f[:, 0, 0]-removed)


GRID = np.array([[0., b, e] for b in (-.3, 0., .3) for e in (-.1, 0., .1)])
BOUNDS = [(.5, 4.), (.5, 4.), (.1, 5.), (8., 55.), (.1, .95), (.1, .95), (.5, 1.)]


def score(v, shaped=False, budgets=Budgets(), grid=GRID):
    info = effective_info(fisher(grid, v, shaped, budgets))
    return float(np.min(info))
