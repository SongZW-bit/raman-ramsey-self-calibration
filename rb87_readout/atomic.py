"""87Rb D1 angular matrix elements. No sympy or fitted branching ratios.
Units Gamma=2*pi*5.7500 MHz. Includes all 8 ground and 8 excited Zeeman states.
"""
import math,numpy as np
from scipy.linalg import expm,expm_frechet

LINE_MHZ=5.7500
GAMMA_MHZ=2*np.pi*LINE_MHZ
EXC_SPLIT=814.5/LINE_MHZ  # 2*A(5P1/2), downloaded Steck table
GROUND_SPLIT=6834.682610904/LINE_MHZ
MU_B=1.3996246/LINE_MHZ  # muB/h / (Gamma/2pi), MHz/G
GG={1:-.50182671,2:.49983642}
GE={1:-.167705,2:.165715}
GROUND=[(f,m) for f in (1,2) for m in range(-f,f+1)]
EXCITED=GROUND.copy();N=16
G1=GROUND.index((1,0));G2=GROUND.index((2,0))

def fact(x):
    if x<-.00001:return float('inf')
    return math.gamma(x+1)
def tri(a,b,c):
    if c>a+b+1e-9 or c<abs(a-b)-1e-9:return 0.
    return math.sqrt(fact(a+b-c)*fact(a-b+c)*fact(-a+b+c)/fact(a+b+c+1))
def w3(a,b,c,ma,mb,mc):
    if abs(ma+mb+mc)>1e-9 or any(abs(m)>j for j,m in [(a,ma),(b,mb),(c,mc)]):return 0.
    pref=(-1)**int(round(a-b-mc))*tri(a,b,c)*math.sqrt(fact(a+ma)*fact(a-ma)*fact(b+mb)*fact(b-mb)*fact(c+mc)*fact(c-mc))
    s=0.
    for z in range(15):
        args=[z,a+b-c-z,a-ma-z,b+mb-z,c-b+ma+z,c-a-mb+z]
        if min(args)>=-1e-9:s+=(-1)**z/math.prod(fact(x) for x in args)
    return pref*s
def w6(a,b,c,d,e,f):
    pref=tri(a,b,c)*tri(a,e,f)*tri(d,b,f)*tri(d,e,c)
    s=0.
    for z in range(15):
        args=[z-a-b-c,z-a-e-f,z-d-b-f,z-d-e-c,a+b+d+e-z,a+c+d+f-z,b+c+e+f-z]
        if min(args)>=-1e-9:s+=(-1)**z*fact(z+1)/math.prod(fact(x) for x in args)
    return pref*s
def dipole(fe,me,fg,mg,q):
    return np.sqrt(2*(2*fe+1)*(2*fg+1))*(-1)**(fe-me+fg+3)*w3(fe,1,fg,-me,q,mg)*w6(.5,fe,1.5,fg,.5,1)

D={q:np.array([[dipole(fe,me,fg,mg,q) for fg,mg in GROUND] for fe,me in EXCITED]) for q in (-1,0,1)}
I=np.eye(N,dtype=complex)
Z=np.diag([-.5 if f==1 else .5 for f,m in GROUND]+[0.]*8)
LD=-1j*(np.kron(I,Z)-np.kron(Z.T,I))

def vec(x):return np.asarray(x).reshape(N*N,order='F')
def mat(x):return np.asarray(x).reshape((N,N),order='F')
def dissipator(c):
    cc=c.conj().T@c
    return np.kron(c.conj(),c)-.5*(np.kron(I,cc)+np.kron(cc.T,I))

def jumps(resolve_excited=False):
    # Resolved final ground F (6.8 GHz photons), same polarization q.
    # Sum F' amplitudes BEFORE dissipating: virtual Raman paths interfere.
    # The full Hamiltonian retains F' splitting and free decay spectral beating.
    result=[]
    for fg in (1,2):
        for fe in ((1,2) if resolve_excited else (None,)):
            for q in (-1,0,1):
                c=np.zeros((N,N),complex)
                for i,(ff,mg) in enumerate(GROUND):
                    for j,(ef,me) in enumerate(EXCITED):
                        if ff==fg and (fe is None or ef==fe):c[i,8+j]=D[q][j,i]
                if np.linalg.norm(c):result.append(c)
    return result
JUMPS=jumps()
DECAY=sum((dissipator(c) for c in JUMPS),np.zeros((N*N,N*N),complex))
# Physical separate final F manifolds are resolved in fluorescence.
GROUPS=[[i for i,(ff,m) in enumerate(GROUND) if ff==f] for f in (1,2)]+[list(range(8,16))]

def ham(delta=0,detuning=0,omega=.1,phase=0,B=.1,ratio=1.,pol=None,cross_shift=True,exc2=True):
    if pol is None:pol={1:1.}
    h=np.zeros((N,N),complex)
    for i,(f,m) in enumerate(GROUND):h[i,i]=delta*(-.5 if f==1 else .5)+GG[f]*MU_B*B*m
    for i,(f,m) in enumerate(EXCITED):h[8+i,8+i]=detuning+(EXC_SPLIT if f==2 else 0)+GE[f]*MU_B*B*m
    # detuning=0 is the F'=1 hyperfine centroid; actual optical Zeeman shifts retained.
    for j,(fe,me) in enumerate(EXCITED):
        for i,(fg,mg) in enumerate(GROUND):
            c=sum(e*D[q][j,i] for q,e in pol.items())
            amp=omega*(ratio*np.exp(1j*phase) if fg==2 else 1.)
            if not exc2 and fe==2:c=0
            h[8+j,i]=amp*c/2;h[i,8+j]=np.conj(h[8+j,i])
            # Other optical colour off-resonantly addresses the same ground F.
            # Leading diagonal ac shift; omitted fast micromotion is audited separately.
            den=detuning+(EXC_SPLIT if fe==2 else 0)+(GROUND_SPLIT if fg==1 else -GROUND_SPLIT)
            if cross_shift and abs(den)>1:
                other=omega*(ratio if fg==1 else 1.)
                h[i,i]-=abs(other*c)**2/(4*den)
    return h

def liouvillian(delta=0,detuning=0,omega=.1,phase=0,gamma=1e-5,B=.1,ratio=1.,pol=None,cross_shift=True,exc2=True,compensate=False):
    if compensate and omega:
        s,_=raman_rates(detuning,omega,ratio,B,pol,cross_shift,exc2)
        delta=delta-s
    h=ham(delta,detuning,omega,phase,B,ratio,pol,cross_shift,exc2)
    c=np.sqrt(2*gamma)*Z # ground coherence between F1 and F2 decays at gamma
    return -1j*(np.kron(I,h)-np.kron(h.T,I))+DECAY+dissipator(c)

def raman_rates(detuning,omega,ratio=1.,B=.1,pol=None,cross_shift=True,exc2=True):
    h=ham(0,detuning,omega,0,B,ratio,pol,cross_shift,exc2)
    v=h[8:,:8];den=np.diag(h)[8:].real
    he=h[:8,:8]-v.conj().T@np.diag(den/(den**2+.25))@v
    stark=(he[G2,G2]-he[G1,G1]).real
    return stark,2*abs(he[G1,G2])

def propagate(r,d,l,t,total=True):
    if t==0:return r.copy(),d.copy()
    if total:e,de=expm_frechet(l*t,LD*t)
    else:e=expm(l*t);de=np.zeros_like(e)
    return mat(e@vec(r)),mat(e@vec(d)+de@vec(r))

def prepare(delta=0,gamma=1e-5,tp_us=20,td_us=100,omega=.1,B=.1,initial='selected'):
    r=np.zeros((N,N),complex)
    if initial=='selected':r[G1,G1]=1
    elif initial=='mixed':r[np.arange(8),np.arange(8)]=1/8
    else:raise ValueError(initial)
    d=np.zeros_like(r)
    r,d=propagate(r,d,liouvillian(delta=delta,omega=omega,gamma=gamma,B=B),tp_us*GAMMA_MHZ)
    r,d=propagate(r,d,liouvillian(delta=delta,omega=0,gamma=gamma,B=B),td_us*GAMMA_MHZ)
    return r,d

def fi(p,dp):
    p=np.real(p);dp=np.real(dp);valid=p>1e-13
    return float(np.sum(dp[valid]**2/p[valid]))
def probs(r,d,eta=1.):
    p=np.array([np.trace(r[np.ix_(g,g)]).real for g in GROUPS]);dp=np.array([np.trace(d[np.ix_(g,g)]).real for g in GROUPS])
    c=np.array([[eta,1-eta,0],[1-eta,eta,0],[0,0,1]])
    return c@p,c@dp
def pop(r,d,eta=1.):return fi(*probs(r,d,eta))
def qfi(r,d):
    va,u=np.linalg.eigh((r+r.conj().T)/2);dd=u.conj().T@d@u;den=va[:,None]+va[None,:];ok=den>1e-12
    return float(2*np.sum(abs(dd[ok])**2/den[ok]))
def coh(r,d,phi=np.pi/2,grouped=True):
    u=I.copy();u[G1,G1]=u[G1,G2]=1/np.sqrt(2);u[G2,G1]=np.exp(1j*phi)/np.sqrt(2);u[G2,G2]=-np.exp(1j*phi)/np.sqrt(2)
    rr=u.conj().T@r@u;dd=u.conj().T@d@u
    return pop(rr,dd) if grouped else fi(np.diag(rr),np.diag(dd))

if __name__=='__main__':
    import time
    print('clock matrix',D[1][:,[G1,G2]])
    print('sum branching per excited',sum(np.sum(x*x,axis=1) for x in D.values()))
    t=time.perf_counter();r,d=prepare();print('elapsed',time.perf_counter()-t,'trace',np.trace(r),'qfi',qfi(r,d),'pop',pop(r,d),'coh',max(coh(r,d,p) for p in np.linspace(0,2*np.pi,41)))
    print('ground populations',np.diag(r).real[:8])
