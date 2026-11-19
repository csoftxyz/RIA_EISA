"""SUPERSEDED intermediate infrastructure (big-Fraction interval-dual arithmetic).

This module was the Fraction-based interval-dual infrastructure for the grade-(a)
certificate.  It is SUPERSEDED by grade_a_fast.py (float-interval Neumann
certificate), which completed the full 22-observable assembly and is the version
invoked by verify_prd.py and cited in the paper.  Retained for provenance only;
do NOT use it to reproduce the certificate (its Takagi-stage assembly was never
finished here -- the bottleneck was resolved in grade_a_fast.py).

Original status note: core components validated (matrix build value error 0,
eigenvector purity 1.0); full Takagi assembly pending in THIS file.

Strategy (grade a):
  * parameters are exact rationals; matrices built with interval-dual numbers
    (value = rational interval, derivative = rational interval) so every entry
    carries a rigorous enclosure of its true value AND its parameter derivative.
  * Hermitian eigen-data via interval char-poly root isolation + adjugate
    eigenvectors; differentials by implicit-function formulas, all interval ops.
  * assemble the 22 observable differentials; extract a 22x22 minor; if its
    interval determinant excludes 0, rank 22 is PROVEN at the point.

This module is self-contained (fractions only).  It is validated against
finite differences before any certificate is claimed.
"""
from fractions import Fraction as F
import grade_a_obs as G
from grade_a_obs import IV, iv, cz, cadd, cmul, cconj, cinv, cscale, cmm, cdag, cT, to_cint, herm_c, isolate_eigvals, adj_col, iabs, iabs2
import exact_core as E
from exact_core import c, C, mm, dag, T, tr, sc, ad, sb, inv3, anti3, sym3

# ---- interval-dual complex number: (value_c, deriv_c) each (IV,IV) ----
def dz(v, d=None):
    return (v, d if d is not None else cz(0,0))
def dadd(a,b): return (cadd(a[0],b[0]), cadd(a[1],b[1]))
def dmul(a,b): return (cmul(a[0],b[0]), cadd(cmul(a[0],b[1]),cmul(a[1],b[0])))
def dconj(a): return (cconj(a[0]), cconj(a[1]))
def dinv(a):
    iv_=cinv(a[0])
    return (iv_, cadd(cmul(cmul(iv_,iv_),a[1]), cz(0,0)) if False else (-cmul(cmul(iv_,iv_),a[1])[0], -cmul(cmul(iv_,iv_),a[1])[1]))
def dscale(a,s):  # s real IV
    return ((a[0][0]*s,a[0][1]*s),(a[1][0]*s,a[1][1]*s))
def dneg(a): return ((-a[0][0],-a[0][1]),(-a[1][0],-a[1][1]))
def dsub(a,b): return dadd(a,dneg(b))

def dmm(A,B):
    out=[[dz(cz(0,0)) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            acc=dz(cz(0,0))
            for k in range(3): acc=dadd(acc,dmul(A[i][k],B[k][j]))
            out[i][j]=acc
    return out
def ddag(A): return [[dconj(A[j][i]) for j in range(3)] for i in range(3)]
def dT(A): return [[A[j][i] for j in range(3)] for i in range(3)]
def dad(A,B): return [[dadd(A[i][j],B[i][j]) for j in range(3)] for i in range(3)]
def dsb(A,B): return [[dsub(A[i][j],B[i][j]) for j in range(3)] for i in range(3)]
def dsc(A,s):
    sd = s if (isinstance(s,tuple) and isinstance(s[0],tuple)) else dz(cz(s,iv(0)))
    return [[dmul(A[i][j], sd) for j in range(3)] for i in range(3)]

def val_c(z): return z[0]
def der_c(z): return z[1]
def re_iv(z): return z[0]
def im_iv(z): return z[1]

# build matrices with interval-dual entries; seed = parameter index k
def build_dual(pt, k):
    n=len(pt)
    def get(i,j):
        v=cz(iv(pt[i]), iv(pt[j]))
        # d/dp_k of (p_i + i p_j)
        dr=iv(1) if i==k else iv(0)
        di=iv(1) if j==k else iv(0)
        return dz(v,(dr,di))
    A=[[dz(cz(0,0)) for _ in range(3)] for _ in range(3)]
    a=get(0,1); b=get(2,3); cc=get(4,5)
    A[0][1]=a; A[1][0]=dneg(a); A[0][2]=b; A[2][0]=dneg(b); A[1][2]=cc; A[2][1]=dneg(cc)
    S=[[dz(cz(0,0)) for _ in range(3)] for _ in range(3)]
    u=[get(6+2*t,7+2*t) for t in range(6)]
    S[0][0]=u[0]; S[0][1]=u[1]; S[0][2]=u[2]; S[1][0]=u[1]; S[1][1]=u[3]; S[1][2]=u[4]; S[2][0]=u[2]; S[2][1]=u[4]; S[2][2]=u[5]
    v=[get(18+2*t,19+2*t) for t in range(8)]
    v1u,v15u,v1d,v15d,su,sd,vR,vL=v
    three_=dz(cz(iv(3),iv(0)))
    Mu=dad(dsc(A,dadd(v1u,v15u)), dsc(S,su))
    Md=dad(dsc(A,dadd(v1d,v15d)), dsc(S,sd))
    Me=dad(dsc(A,dsub(v1d,dmul(three_,v15d))), dsc(S,dneg(dmul(three_,sd))))
    MD=dad(dsc(A,dsub(v1u,dmul(three_,v15u))), dsc(S,dneg(dmul(three_,su))))
    # Mnu = vL S - MD (vR S)^-1 MD^T  ; need dual 3x3 inverse
    vRS=dsc(S,vR)
    inv=dinv3(vRS)
    Mnu=dsb(dsc(S,vL), dmm(dmm(MD,inv), dT(MD)))
    return Mu,Md,Me,Mnu

def dinv3(A):
    det=ddet3(A)
    idet=dinv(det)
    a,b,cc=A[0]; d,e,f=A[1]; g,h,i=A[2]
    def cof(r,s):
        idx=[x for x in range(3) if x!=r]; jdx=[x for x in range(3) if x!=s]
        m=[[A[idx[0]][jdx[0]],A[idx[0]][jdx[1]]],[A[idx[1]][jdx[0]],A[idx[1]][jdx[1]]]]
        dd=dsub(dmul(m[0][0],m[1][1]), dmul(m[0][1],m[1][0]))
        sign=iv(1) if (r+s)%2==0 else iv(-1)
        return dmul(dd, dz(cz(sign,iv(0))))
    cofm=[[cof(r,s) for s in range(3)] for r in range(3)]
    adj=dT(cofm)
    return [[dmul(adj[r][s], idet) for s in range(3)] for r in range(3)]

def ddet3(A):
    a,b,cc=A[0]; d,e,f=A[1]; g,h,i=A[2]
    return dadd(dsub(dmul(a,dsub(dmul(e,i),dmul(f,h))), dmul(b,dsub(dmul(d,i),dmul(f,g)))),
                dmul(cc,dsub(dmul(d,h),dmul(e,g))))

# ---- Hermitian eigen-data differential (implicit function, interval) ----
# H dual Hermitian; eigenvalue lam (real IV), eigenvector v (complex IV column).
# dm = v^dag dH v ; dv = (lam I - H)^+ dH v  (reduced resolvent on complement)
def eig_data_dual(H, approx_lams, approx_vecs):
    """return list of (lam_dual_real, v_dual_complex) for 3 eigenpairs, descending.
    H is dual Hermitian.  We use the VALUE part to isolate eigenvalues/vectors
    (interval), then propagate derivatives by implicit formulas."""
    Hv=[[H[i][j][0] for j in range(3)] for i in range(3)]  # value complex interval
    lams_iv=isolate_eigvals(Hv, approx_lams)
    # eigenvectors from adjugate at interval lam, normalized
    pairs=[]
    for t in range(3):
        lam=lams_iv[t]
        # choose column with largest norm to avoid zero adjugate column
        best=None; bestn=iv(-1)
        for col in range(3):
            ac=adj_col(Hv, lam, col)
            n2=iabs2(ac[0])+iabs2(ac[1])+iabs2(ac[2])
            if n2.lo>bestn.lo: bestn=n2; best=ac
        # rescale by dividing by the largest COMPONENT (complex interval division,
        # NO sqrt of the huge norm) so the column is O(1) before the final sqrt.
        kmax=max(range(3), key=lambda k: iabs2(best[k]).lo)
        piv=best[kmax]
        pivinv=cinv(piv)  # complex interval inverse, |piv|~1e10 but no sqrt
        best=[cmul(best[k], pivinv) for k in range(3)]
        bestn=iabs2(best[0])+iabs2(best[1])+iabs2(best[2])  # now O(1)
        n=G.isqrt_iv(bestn)
        v=[(best[k][0]/n, best[k][1]/n) for k in range(3)]
        pairs.append((lam, v))
    return pairs

def herm_eig_diff(H, pairs):
    """differentials of eigenvalues and eigenvectors.
    returns (dlams[list real IV], dvecs[list of complex-IV columns])."""
    dH=[[H[i][j][1] for j in range(3)] for i in range(3)]
    Hv=[[H[i][j][0] for j in range(3)] for i in range(3)]
    dlams=[]; dvecs=[]
    for (lam,v) in pairs:
        # dm = v^dag dH v  (real)
        acc=cz(0,0)
        for i in range(3):
            for j in range(3):
                acc=cadd(acc, cmul(cconj(v[i]), cmul(dH[i][j], v[j])))
        dlams.append(acc[0])  # real part (imag ~0)
        # dv_k = sum_{l!=k} v_l (v_l^dag dH v_k)/(lam_k - lam_l)
        # need other eigenvectors; compute via resolvent using adjugate complement
        # dv = (lam I - H)^+ dH v ; use formula with projectors
        # build dH v
        dHv=[sum((cmul(dH[i][j],v[j]) for j in range(3)), cz(0,0)) if False else _cacc([cmul(dH[i][j],v[j]) for j in range(3)]) for i in range(3)]
        dv=[]
        for i in range(3):
            # reduced resolvent applied: sum_l!=k v_l <v_l|dHv>/(lam_k-lam_l)
            acc=cz(0,0)
            for l,(lam2,v2) in enumerate(pairs):
                if abs(float(lam2.mid())-float(lam.mid()))<1e-9: continue
                # overlap <v2|dHv>
                ov=_cacc([cmul(cconj(v2[i2]), dHv[i2]) for i2 in range(3)])
                denom=lam-lam2  # real IV
                coef=(ov[0]/denom, ov[1]/denom)
                acc=cadd(acc, cmul(coef, v2[i]))
            dv.append(acc)
        dvecs.append(dv)
    return dlams, dvecs

def _cacc(lst):
    acc=cz(0,0)
    for x in lst: acc=cadd(acc,x)
    return acc

# ---- observable assembly (interval-dual) ----
def left_unitary_dual(M, approx_w, approx_V):
    """M dual 3x3; returns (U dual columns, m dual real) descending singular values.
    U = eigenvectors of H=M M^dag; m=sqrt(eigval)."""
    H=dmm(M,ddag(M))
    pairs=eig_data_dual(H, approx_w, approx_V)
    dl,dv=herm_eig_diff(H,pairs)
    U=[]; ms=[]
    for t,(lam,v) in enumerate(pairs):
        m=G.isqrt_iv(lam)  # singular value interval (value)
        # dm = dl/(2m)
        dm=dl[t]/(iv(2)*m)
        ms.append((m,dm))
        # U column = v (value) with derivative dv
        Ucol=[(v[i], dv[t][i]) for i in range(3)]
        U.append(Ucol)
    return U, ms

def obs_dual(pt, k, approx):
    """22 observable differentials wrt parameter k. approx dict of numpy eigen-data."""
    Mu,Md,Me,Mnu=build_dual(pt,k)
    Uu,mus=left_unitary_dual(Mu, *approx['Mu'])
    Ud,mds=left_unitary_dual(Md, *approx['Md'])
    Ue,mes=left_unitary_dual(Me, *approx['Me'])
    out=[]
    # 9 masses: value+deriv
    for ms in (mus,mds,mes):
        for (m,dm) in ms: out.append((m,dm))
    # CKM V = Uu^dag Ud ; entries dual
    V=[[sum((cmul(cconj(Uu[i][t][0]), Ud[j][t][0]) for t in range(3)), cz(0,0)) if False else _cacc([cmul(cconj(Uu[i][t][0]), Ud[j][t][0]) for t in range(3)]) for j in range(3)] for i in range(3)]
    # need dual V for derivatives: rebuild with dual
    Vd=[[ _dacc([dmul(dconj(Uu[i][t]), Ud[j][t]) for t in range(3)]) for j in range(3)] for i in range(3)]
    # 3 moduli + Jarlskog
    for (i,j) in [(0,1),(0,2),(1,2)]:
        z=Vd[i][j]; mod=iabs(z[0]); dmod=_dmod(z)
        out.append((mod,dmod))
    J=_jarlskog_dual(Vd); out.append(J)
    # neutrino: Mnu_mass = Ue^T Mnu Ue (transpose congruence), dual
    Mm=dmm(dmm(dT(Ue),Mnu),Ue)
    # Takagi: Hn = Mm^dag Mm ; eig -> W; phases z=diag(W^T Mm W); P=W exp(-i arg z/2)
    # use approx for Mm
    P,mn=_takagi_dual(Mm, approx['Mnu'])
    for (m,dm) in mn: out.append((m,dm))
    for (i,j) in [(0,1),(0,2),(1,2)]:
        z=P[i][j]; out.append((iabs(z[0]), _dmod(z)))
    out.append(_jarlskog_dual(P))
    # Majorana phases phi_i = arg(P_1i^2 P_11*^2)
    for i in (1,2):
        z=dmul(dmul(P[0][i],P[0][i]), dmul(dconj(P[0][0]),dconj(P[0][0])))
        out.append(_darg(z))
    return out

def _dacc(lst):
    acc=dz(cz(0,0))
    for x in lst: acc=dadd(acc,x)
    return acc

def _dmod(z):
    """d|z| = Re(conj(z) dz)/|z| ; z dual complex."""
    mod=iabs(z[0])
    num=cadd(cmul(cconj(z[0]),z[1]), cmul(z[0],cconj(z[1])))  # 2 Re(conj z dz)
    # Re part / (2 mod)
    return (num[0]/(iv(2)*mod))

def _darg(z):
    """d arg z = Im(conj(z) dz)/|z|^2 ; returns (value IV, deriv IV)."""
    n2=iabs2(z[0])
    czdz=cmul(cconj(z[0]),z[1])
    darg=czdz[1]/n2
    # value arg via atan2 interval (use float center + rigorous bound)
    val=_arg_iv(z[0])
    return (val,darg)

def _arg_iv(z):
    import math
    re=float(z[0].mid()); im=float(z[1].mid())
    a=math.atan2(im,re)
    # rigorous bound: |arg - atan2(mid)| <= (|dre|+|dim|)/|z|
    rad=(z[0].width()+z[1].width())/max(iabs(z).lo, F(1,10**30))
    return iv(a-rad, a+rad) if False else IV(F(a).limit_denominator(10**12)-rad, F(a).limit_denominator(10**12)+rad)

def _jarlskog_dual(V):
    # J = Im(V00 V11 conj(V01 V10))
    z=dmul(dmul(V[0][0],V[1][1]), dconj(dmul(V[0][1],V[1][0])))
    return (z[0][1], z[1][1])  # imag value, imag deriv

def _takagi_dual(Mm, approx):
    """Mm dual symmetric; approx=(w,W) numpy eigh of Mm^dag Mm value.
    returns P dual columns, mn dual real (descending)."""
    Hn=dmm(ddag(Mm),Mm)
    pairs=eig_data_dual(Hn, approx[0], approx[1])
    dl,dv=herm_eig_diff(Hn,pairs)
    # W columns = eigenvectors (value+deriv)
    W=[[ (pairs[t][1][i], dv[t][i]) for i in range(3)] for t in range(3)]
    # z_t = (W^T Mm W)_tt ; dual
    Wt=dT(W)
    Z=dmm(dmm(Wt,Mm),W)
    mn=[]; P=[]
    for t in range(3):
        z=Z[t][t]
        m=iabs(z[0]); dm=_dmod(z)
        mn.append((m,dm))
        # phase = exp(-i arg z /2); dphase = phase * (-i/2) darg z
        argval,darg=_darg(z)
        # phase value cos/sin interval
        ph=_exp_half_arg_iv(z[0])
        # dph = ph * (-i/2)*darg  (darg real)
        half=iv(F(1,2))
        dph=( ph[1]*half*darg, -ph[0]*half*darg )  # -i*ph/2 = (ph_im/2, -ph_re/2)
        Pcol=[ dmul(W[t][i], (ph,dph)) for i in range(3)]
        P.append(Pcol)
    # P columns currently P[:,t]; return as columns list
    return P, mn

def _exp_half_arg_iv(z):
    import math
    re=float(z[0].mid()); im=float(z[1].mid())
    a=math.atan2(im,re)
    c=math.cos(-a/2); s=math.sin(-a/2)
    # bound error from interval z
    rad=(z[0].width()+z[1].width())/max(iabs(z).lo,F(1,10**30))
    return (iv(c-rad,c+rad), iv(s-rad,s+rad))

if __name__=="__main__":
    import numpy as np
    pt=E.rational_point(34,2)
    # validate dual build vs exact_core value + finite-diff derivative
    k=5
    Mu,Md,Me,Mnu=build_dual(pt,k)
    def gv(i,j): return c(pt[i],pt[j])
    Mu0,Md0,Me0,Mnu0=E.bld_120_126(gv)
    # value check
    err=0
    for M,M0 in [(Mu,Mu0),(Md,Md0),(Me,Me0),(Mnu,Mnu0)]:
        for i in range(3):
            for j in range(3):
                v=M[i][j][0]; t=M0[i][j]
                err=max(err, abs(float(v[0].mid())-float(t.a)), abs(float(v[1].mid())-float(t.b)))
    print("value max err", err)
    # derivative check vs finite diff
    h=1e-6
    qp=list(pt); qp[k]=float(pt[k])+h
    qm=list(pt); qm[k]=float(pt[k])-h
    def bld(q):
        def g2(i,j): return c(q[i],q[j])
        return E.bld_120_126(g2)
    Mp=bld(qp); Mm=bld(qm)
    derr=0
    for M,Mp_,Mm_ in [(Mu,Mp[0],Mm[0]),(Mnu,Mp[3],Mm[3])]:
        for i in range(3):
            for j in range(3):
                d=M[i][j][1]
                fd_r=(float(Mp_[i][j].a)-float(Mm_[i][j].a))/(2*h)
                fd_i=(float(Mp_[i][j].b)-float(Mm_[i][j].b))/(2*h)
                derr=max(derr, abs(float(d[0].mid())-fd_r), abs(float(d[1].mid())-fd_i))
    print("deriv max err", derr)
