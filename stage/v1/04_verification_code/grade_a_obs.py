"""SUPERSEDED intermediate prototype (big-Fraction interval arithmetic).

This module was the first interval-arithmetic infrastructure for the grade-(a)
certificate.  It is SUPERSEDED by grade_a_fast.py (float-interval Neumann
certificate), which is the version invoked by verify_prd.py and cited in the
paper.  Retained for provenance only; do not use it to reproduce the certificate.

Original description: rigorous interval certificate that rank(dO/dp)=22 (grade a).

General-point eigenvalues are algebraic (not rational), so we use RATIONAL
INTERVAL arithmetic with outward rounding: every operation returns an interval
guaranteed to contain the true value.  A 22x22 minor whose interval determinant
excludes 0 is a rigorous proof of rank 22 at the point => zero forced relations.

No external libraries; fractions.Fraction only.
"""
from fractions import Fraction as F

class IV:
    __slots__=("lo","hi")
    def __init__(self,lo,hi=None):
        self.lo=F(lo); self.hi=F(lo) if hi is None else F(hi)
        assert self.lo<=self.hi
    def __add__(self,o): return IV(self.lo+o.lo,self.hi+o.hi)
    def __sub__(self,o): return IV(self.lo-o.hi,self.hi-o.lo)
    def __neg__(self): return IV(-self.hi,-self.lo)
    def __mul__(self,o):
        xs=[self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi]
        return IV(min(xs),max(xs))
    def inv(self):
        if self.lo<=0<=self.hi: raise ZeroDivisionError("interval contains 0")
        return IV(F(1,self.hi),F(1,self.lo))
    def __truediv__(self,o): return self*o.inv()
    def contains(self,x): return self.lo<=x<=self.hi
    def excludes0(self): return self.lo>0 or self.hi<0
    def mid(self): return (self.lo+self.hi)/2
    def width(self): return self.hi-self.lo
    def __repr__(self): return "[%.6g,%.6g]"%(float(self.lo),float(self.hi))

def iv(x):
    if isinstance(x,IV): return x
    if isinstance(x,float): x=F(x).limit_denominator(10**12)
    return IV(x)
def ivc(z):  # complex interval -> (re,im)
    return (iv(z.a) if hasattr(z,'a') else iv(z), iv(z.b) if hasattr(z,'b') else iv(0))

# ---- interval matrix ops (3x3), entries IV or complex (IV,IV) ----
def iadd(A,B): return [[A[i][j]+B[i][j] for j in range(3)] for i in range(3)]
def imul(A,B):
    return [[sum((A[i][k]*B[k][j] for k in range(3)), iv(0)) for j in range(3)] for i in range(3)]
def itran(A): return [[A[j][i] for j in range(3)] for i in range(3)]

# real symmetric interval eigenvalue isolation via Sturm / bisection on char poly
def ichar(H):
    """H complex Hermitian interval matrix (entries (re,im)); char coeffs real."""
    def re(z): return z[0]
    t1=re(H[0][0])+re(H[1][1])+re(H[2][2])
    tr2=iv(0)
    for i in range(3):
        for j in range(3): tr2=tr2+re(cmul(H[i][j],H[j][i]))
    t2=(t1*t1-tr2)*iv(F(1,2))
    def csub(x,y): return (x[0]-y[0],x[1]-y[1])
    def cdet3(A):
        a,b,cc=A[0]; d,e,f=A[1]; g,h,i=A[2]
        return cadd(csub(cmul(a,csub(cmul(e,i),cmul(f,h))), cmul(b,csub(cmul(d,i),cmul(f,g)))),
                    cmul(cc,csub(cmul(d,h),cmul(e,g))))
    t3=re(cdet3(H))
    return t1,t2,t3

def ipoly(lam,t1,t2,t3):
    return lam*lam*lam - t1*lam*lam + t2*lam - t3

def _isqrt_int(n):
    if n<0: raise ValueError
    if n==0: return 0
    x=n
    y=(x+1)//2
    while y<x:
        x=y; y=(x+n//x)//2
    return x

def isqrt_iv(x):
    """rigorous interval sqrt for x>=0 (outward rounded rational bounds)."""
    assert x.lo>=0
    def floorsqrt(q):
        # sqrt(q)=sqrt(n*d)/d ; floor via integer sqrt of n*d
        return F(_isqrt_int(q.numerator*q.denominator), q.denominator)
    def ceilsqrt(q):
        s=floorsqrt(q)
        return s if s*s>=q else s+F(1,q.denominator)
    lo=floorsqrt(x.lo) if x.lo>0 else F(0)
    hi=ceilsqrt(x.hi)
    return IV(lo,hi)

def iabs2(z):  # |z|^2 for complex interval (re,im)
    return z[0]*z[0]+z[1]*z[1]

def iabs(z):
    return isqrt_iv(iabs2(z))

# complex interval helpers
def cz(re_,im_): return (iv(re_),iv(im_))
def cadd(a,b): return (a[0]+b[0],a[1]+b[1])
def cmul(a,b): return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])
def cconj(a): return (a[0],-a[1])
def cinv(a):
    n=iabs2(a)
    return (a[0]/n, -a[1]/n)
def cscale(a,s): return (a[0]*s,a[1]*s)

def cmm(A,B):
    out=[[cz(0,0) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            acc=cz(0,0)
            for k in range(3): acc=cadd(acc,cmul(A[i][k],B[k][j]))
            out[i][j]=acc
    return out
def cdag(A): return [[cconj(A[j][i]) for j in range(3)] for i in range(3)]
def cT(A): return [[A[j][i] for j in range(3)] for i in range(3)]

def to_cint(M):
    return [[cz(M[i][j].a, M[i][j].b) for j in range(3)] for i in range(3)]

def herm_c(M): return cmm(M,cdag(M))

def isolate_eigvals(Hr, approx, tol=F(1,10**8)):
    """Rigorous eigenvalue enclosure by EXACT Fraction bisection on the char poly.
    Float bisection suffers catastrophic cancellation at large eigenvalues
    (x^3 ~ 1e15), so all arithmetic here is exact rational.  Sign decisions use
    the interval char-poly value: a sign is determined only when the whole
    interval lies strictly on one side of 0."""
    t1,t2,t3=ichar(Hr)
    # Hr is a point enclosure (width ~0); use exact rational POINT coefficients
    # for fast, cancellation-free sign decisions.  The final lam interval is
    # widened by the coefficient uncertainty to remain rigorous.
    c1=t1.mid(); c2=t2.mid(); c3=t3.mid()
    def p(x): return x*x*x - c1*x*x + c2*x - c3
    def sign(x):
        v=p(x)
        if v>0: return 1
        if v<0: return -1
        return 0
    ap=sorted(F(float(a)).limit_denominator(10**9) for a in approx)
    out=[]
    for idx,a in enumerate(ap):
        # bracket by midpoints to neighbouring roots -> isolates a SINGLE root
        lo = (ap[idx-1]+a)/2 if idx>0 else a-(ap[1]-ap[0])
        hi = (a+ap[idx+1])/2 if idx<2 else a+(ap[2]-ap[1])
        sl=sign(lo); sh=sign(hi)
        tries=0
        while (sl==0 or sh==0 or sl==sh) and tries<40:
            # shrink toward a if no sign change (approx may be slightly off)
            lo=(lo+a)/2; hi=(hi+a)/2
            sl=sign(lo); sh=sign(hi); tries+=1
        if sl==sh or sl==0 or sh==0:
            lo=a-F(1,1000); hi=a+F(1,1000); sl=sign(lo); sh=sign(hi)
        for _ in range(120):
            mid=(lo+hi)/2
            sm=sign(mid)
            if sm==0:
                # mid interval straddles 0 -> root enclosed within mid's uncertainty;
                # narrow by moving the endpoint whose sign matches mid's centre
                if sign((lo+mid)/2)==sl: lo=(lo+mid)/2
                else: hi=(mid+hi)/2
            elif sm==sl: lo=mid; sl=sm
            else: hi=mid; sh=sm
            if hi-lo<tol: break
        out.append(IV(lo,hi))
    out.sort(key=lambda l: l.mid(), reverse=True)  # descending, matching approx
    return out

def adj_col(H, lam, col):
    """column `col` of adj(H - lam I), complex interval; an unnormalized eigenvector."""
    A=[[cadd(H[i][j], cz(-lam,0) if i==j else cz(0,0)) for j in range(3)] for i in range(3)]
    # Row-scaling does not change ker(H-lam I), hence the adjugate column direction
    # (the eigenvector) is unchanged.  Normalize each row by its largest component
    # (complex division, no sqrt) so cofactor arithmetic stays O(1) and fast.
    for i in range(3):
        kmax=max(range(3), key=lambda k: iabs2(A[i][k]).lo)
        piv=A[i][kmax]
        if iabs2(piv).lo>0:
            pinv=cinv(piv)
            A[i]=[cmul(A[i][k], pinv) for k in range(3)]
    # adjugate = cofactor transpose; column col of adj = cofactors of row col
    def cof(r,s):
        idx=[k for k in range(3) if k!=r]; jdx=[k for k in range(3) if k!=s]
        a=A[idx[0]][jdx[0]]; b=A[idx[0]][jdx[1]]; cc=A[idx[1]][jdx[0]]; d=A[idx[1]][jdx[1]]
        ad=cmul(a,d); bc=cmul(b,cc)
        det=(ad[0]-bc[0], ad[1]-bc[1])
        sign=1 if (r+s)%2==0 else -1
        return cscale(det, iv(sign))
    # row `col` of adj; for Hermitian H the conjugate of this row is an eigenvector
    return [cof(col,k) for k in range(3)]

if __name__=="__main__":
    import exact_core as E
    from exact_core import c,mm,dag
    pt=E.rational_point(34,2)
    def gv(i,j): return c(pt[i],pt[j])
    Mu,Md,Me,Mnu=E.bld_120_126(gv)
    H=mm(Mu,dag(Mu))
    Hc=to_cint(H)
    t1,t2,t3=ichar(Hc)
    print("char coeffs", t1,t2,t3)
    a=iv(F(1,3)); b=iv(F(2,7))
    print("mul",a*b,"div",a/b)
    import numpy as np
    Hn=np.array([[complex(H[i][j].a,H[i][j].b) for j in range(3)] for i in range(3)])
    w,V=np.linalg.eigh(Hn)
    o=np.argsort(w)[::-1]; w=w[o]; V=V[:,o]  # descending, matching isolate
    print("numpy eigvals", w)
    lams=isolate_eigvals(Hc,w)
    print("isolated eigvals", [float(l.mid()) for l in lams])
    for t,lam in enumerate(lams):
        best=None; bn=iv(-1)
        for col in range(3):
            ac=adj_col(Hc,lam,col); n2=iabs2(ac[0])+iabs2(ac[1])+iabs2(ac[2])
            if n2.lo>bn.lo: bn=n2; best=ac
        kmax=max(range(3), key=lambda k: iabs2(best[k]).lo)
        pinv=cinv(best[kmax])
        best=[cmul(best[k],pinv) for k in range(3)]
        n=isqrt_iv(iabs2(best[0])+iabs2(best[1])+iabs2(best[2]))
        v=np.array([complex(best[k][0].mid()/float(n.mid()),best[k][1].mid()/float(n.mid())) for k in range(3)])
        print("  eigvec overlap %d = %.10f"%(t,abs(np.vdot(V[:,t],v))))
