"""All 16 87Rb D1 states, two circular Raman fields, exact Lindblad dynamics.

The sensor pair is |5S1/2,F=1,m=-1> and |F=1,m=+1>. A rotation
at half the optical frequency difference makes BOTH sigma+ and sigma-
fields stationary, including their coupling to every allowed D1 transition.
There is no ground-manifold-selective optical-coupling approximation.
Frequency units are 2*pi*unit_mhz. D2 states are not included.
"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import numpy as np
from scipy.linalg import expm
from two_beam import geometry, ratio_matrix

_spec = importlib.util.spec_from_file_location("rb87_d1_atomic", Path(__file__).resolve().parents[1]/"rb87_readout"/"atomic.py")
atomic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atomic)
GROUND, EXCITED, D = atomic.GROUND, atomic.EXCITED, atomic.D
N = 16
G1, G2 = GROUND.index((1,-1)), GROUND.index((1,1))
I = np.eye(N)


class RamanD1:
    def __init__(self, detuning_mhz=1000., unit_mhz=.01, field_gauss=.5,
                 gamma_dark=.01, emission=True, decay_resolution="unresolved_excited",
                 minor_helicity=(0.,0.)):
        self.detuning_mhz = detuning_mhz
        self.unit_mhz = unit_mhz
        self.field_gauss = field_gauss
        self.gamma_dark = gamma_dark
        self.emission = emission
        if decay_resolution not in ("excited_hyperfine", "unresolved_excited"):
            raise ValueError("Unknown spontaneous-emission resolution")
        self.decay_resolution = decay_resolution
        if len(minor_helicity)!=2 or any(not 0<=x<1 for x in minor_helicity):
            raise ValueError("Minor-helicity intensity fractions must lie in [0,1)")
        self.minor_helicity = tuple(float(x) for x in minor_helicity)
        self.m = np.array([m for f,m in GROUND+EXCITED],float)
        mu_b = 1.3996246
        rotation_mhz = atomic.GG[1]*mu_b*field_gauss
        ground = [(6834.682610904 if f==2 else 0.) +
                  atomic.GG[f]*mu_b*field_gauss*m-rotation_mhz*m for f,m in GROUND]
        excited = [detuning_mhz+(814.5 if f==2 else 0.)+
                   atomic.GE[f]*mu_b*field_gauss*m-rotation_mhz*m for f,m in EXCITED]
        self.bare = np.array(ground+excited)/unit_mhz
        denominators=self.bare[8:,None]-self.bare[None,:8]
        self.wrong_plus=np.sum(abs(D[1])**2/denominators,axis=0)
        self.wrong_minus=np.sum(abs(D[-1])**2/denominators,axis=0)
        # Group amplitudes for indistinguishable off-resonant scattering
        # pathways. Resolving bare F' is retained as a sensitivity model,
        # not inferred solely from splitting/linewidth for driven Raman light.
        jumps = atomic.jumps(resolve_excited=decay_resolution == "excited_hyperfine")
        self.decay = (sum((atomic.dissipator(c) for c in jumps),
                          np.zeros((N*N,N*N),complex))*(5.7500/unit_mhz)
                      if emission else np.zeros((N*N,N*N),complex))
        # Normalize intensities using ONLY an analytic second-order nominal
        # mapping; all reported populations use the 16-state propagator.
        den = self.bare[8:]
        splus = -D[1].T @ np.diag(1/den) @ D[1]/4
        sminus = -D[-1].T @ np.diag(1/den) @ D[-1]/4
        cross = -D[1].T @ np.diag(1/den) @ D[-1]/4
        self.ap = float(splus[G1,G1]-splus[G2,G2])
        self.am = float(sminus[G1,G1]-sminus[G2,G2])
        self.cross = float(cross[G1,G2])
        if not self.ap*self.am < 0 or abs(self.cross) < 1e-14:
            raise ValueError("No usable differential shift/coupling mapping at this detuning")
        self.field_product = 1/(2*abs(self.cross))
        self.initial = np.zeros((N,N),complex)
        self.initial[G1,G1] = 1.
        # Circular fields preserve parity of m + (excited-state indicator).
        # Emission maps parity-diagonal density blocks into parity-diagonal
        # blocks. Restricting Liouville space to these two blocks is exact.
        parity = (self.m.astype(int) + np.r_[np.zeros(8,int),np.ones(8,int)]) % 2
        self.active = np.flatnonzero((parity[:,None] == parity[None,:]).reshape(-1,order="F"))

    def fields(self, b, power, q=0., common=0., difference=0.):
        i1, i2 = 1+common+difference, 1+common-difference
        if min(i1,i2) < 0:
            raise ValueError("Negative intensity")
        target = b/self.field_product
        roots = np.roots([self.ap,-target,self.am])
        x = next(float(r.real) for r in roots if abs(r.imag)<1e-10 and r.real>0)
        # q is a known physical log-intensity-ratio offset, not copied from
        # the single-excited-state light-shift formula.
        return (np.sqrt(power*self.field_product*x*np.exp(q)*i1),
                np.sqrt(power*self.field_product/x*np.exp(-q)*i2))

    def hamiltonian(self, delta, b, power, phase=0., frequency=0., q=0., common=0., difference=0.):
        h = np.diag(self.bare-(delta-frequency)*self.m/2).astype(complex)
        e1,e2 = self.fields(b,power,q,common,difference)
        minor1,minor2=self.minor_helicity
        if minor1 or minor2:
            h[np.arange(8),np.arange(8)] -= (minor1*e1**2*self.wrong_minus+
                                              minor2*e2**2*self.wrong_plus)/4
        # H_eff[g1,g2] = |C| E1 E2 exp(-i phase).
        relative = np.sign(self.cross)*np.exp(-1j*phase)
        coupling = (e1*np.sqrt(1-minor1)*D[1]+e2*np.sqrt(1-minor2)*relative*D[-1])/2
        h[8:,:8] = coupling
        h[:8,8:] = coupling.conj().T
        return h

    def generator(self, h, dark=False):
        l = -1j*(np.kron(I,h)-np.kron(h.T,I))+self.decay
        if dark and self.gamma_dark:
            # L=sqrt(gamma/2)*m gives rate gamma between m=-1,+1.
            z = np.sqrt(self.gamma_dark/2)*np.diag(self.m)
            l = l+atomic.dissipator(z)
        return l

    def propagate(self, rho, h, duration, dark=False, reduced=True):
        l = self.generator(h,dark)
        v = rho.reshape(-1,order="F")
        if reduced:
            ids = self.active
            result = np.zeros_like(v)
            result[ids] = expm(l[np.ix_(ids,ids)]*duration) @ v[ids]
        else:
            result = expm(l*duration) @ v
        return result.reshape((N,N),order="F")

    @lru_cache(maxsize=768)
    def pulse_map(self, delta,b,power,frequency,q,common,difference,duration):
        h=self.hamiltonian(delta,b,power,0.,frequency,q,common,difference)
        l=self.generator(h)
        return expm(l[np.ix_(self.active,self.active)]*duration)

    def phase_vector(self,phase):
        # Gauge rotation generates both optical phases without a new expm.
        u=np.exp(1j*phase*(self.m-np.r_[np.zeros(8),np.ones(8)])/2)
        return (u[:,None]*u.conj()[None,:]).reshape(-1,order="F")[self.active]

    def metadata(self):
        return dict(species="87Rb",line="D1",states=16,sensor_states=["F=1,m=-1","F=1,m=+1"],
                    detuning_mhz=self.detuning_mhz,unit_mhz=self.unit_mhz,field_gauss=self.field_gauss,
                    spontaneous_emission=self.emission,readout="projector onto F=1,m=+1; all other states count as failure",
                    decay_resolution=self.decay_resolution,
                    minor_helicity=self.minor_helicity,
                    liouville_dimension=len(self.active),D2_included=False,
                    approximations=["optical rotating wave", "linear Zeeman energies", "Markov vacuum decay",
                                    "ideal state preparation and resolved population detection", "no motion"])


def probability_d1(delta,b,noise,v,kind,flags,phases,config,substeps=1,model=None,return_density=False):
    if model is None:
        model = RamanD1(**config.get("atomic_model",{}))
    noise = np.asarray(noise,float)
    ns,k,channels,total = noise.shape
    assert channels==2 and total==6*substeps
    extra = 10 if kind in ("shape","flex") else 6
    frequencies = np.broadcast_to(v[extra]+np.asarray(config.get("frequency_offsets",0.)),(k,))
    delta = np.broadcast_to(np.asarray(delta).reshape(-1,1),(ns,k))
    b = np.broadcast_to(np.asarray(b).reshape(-1,1),(ns,k))
    q = ratio_matrix(v,flags,config)
    _,angles,durations = geometry(v,kind,flags,config,substeps)
    offsets = np.asarray(config.get("segment_phase_offsets",np.zeros((k,6))))
    populations = np.empty((ns,k)); densities = np.empty((ns,k,N,N),complex) if return_density else None
    cache={}
    dark_cache={}
    for sample in range(ns):
        for shot in range(k):
            state=model.initial.reshape(-1,order="F")[model.active].copy()
            for s,angle in enumerate(angles):
                second = int(s>=total//2)
                phase = angle+offsets[shot,s//substeps]
                if second:
                    phase += phases[shot]+v[5]+v[extra+1]*(not flags[shot])
                common,difference = noise[sample,shot,:,s]
                key=(float(delta[sample,shot]),float(b[sample,shot]),float(v[4]),float(frequencies[shot]),
                     float(q[shot,second]),float(common),float(difference),float(durations[s]))
                if key not in cache:
                    if len(cache)>=512:
                        cache.clear()
                    cache[key]=model.pulse_map(*key)
                gauge=model.phase_vector(phase)
                state=gauge*(cache[key]@(gauge.conj()*state))
                if s==total//2-1:
                    duration=v[3] if flags[shot] else v[2]
                    dark_key=(float(delta[sample,shot]),float(duration))
                    if dark_key not in dark_cache:
                        h = np.diag(model.bare-delta[sample,shot]*model.m/2)
                        l=model.generator(h,dark=True)
                        dark_cache[dark_key]=expm(l[np.ix_(model.active,model.active)]*duration)
                    state=dark_cache[dark_key]@state
            rho_vec=np.zeros(N*N,complex)
            rho_vec[model.active]=state
            rho=rho_vec.reshape((N,N),order="F")
            populations[sample,shot] = rho[G2,G2].real
            if return_density:
                densities[sample,shot] = rho
    if return_density:
        return densities
    if np.min(populations)<-1e-7 or np.max(populations)>1+1e-7:
        raise ValueError("Nonphysical population")
    return np.clip(populations,1e-10,1-1e-10)
