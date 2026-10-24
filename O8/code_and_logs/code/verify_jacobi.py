import numpy as np
n=4
def stats(tracelessB):
    rng=np.random.default_rng(0)
    def B_elem():
        H=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); M=H+H.conj().T
        return M if not tracelessB else M-np.trace(M)/4*np.eye(4)
    def br(x,y,p,q):
        kx,vx=x; ky,vy=y
        if kx=='B' and ky=='B': return ('B', vx@vy-vy@vx)
        if kx=='B' and ky=='F': return ('F', vx@vy)         # A v
        if kx=='F' and ky=='B': return ('F', -vy@vx)        # [F,B]=-[B,F]=-B f
        if kx=='B' and ky=='Z': return ('Z', -vx.T@vy)      # [A,f]=-A^T f  (dual, col)
        if kx=='Z' and ky=='B': return ('Z',  vy.T@vx)      # [f,A]=+A^T f
        if kx=='F' and ky=='F': return ('Z', np.zeros(n,complex))
        if kx=='Z' and ky=='Z': return ('F', np.zeros(n,complex))
        v = vx if kx=='F' else vy; f = vy if kx=='F' else vx
        return ('B', p*np.outer(v,f)+q*np.dot(f,v)*np.eye(n))
    def J(x,y,z,p,q):
        pxy=1 if (x[0] in 'FZ' and y[0] in 'FZ') else 0
        return (br(x,br(y,z,p,q),p,q)[1]-br(br(x,y,p,q),z,p,q)[1])-((-1)**pxy)*br(y,br(x,z,p,q),p,q)[1]
    triples=[('B','B','B'),('B','B','F'),('B','B','Z'),('B','F','Z'),('F','F','Z'),('F','Z','Z'),
             ('F','Z','F'),('Z','F','F'),('Z','F','Z')]
    rows=[]
    for _ in range(10):
        for t in triples:
            x=('B',B_elem()) if t[0]=='B' else (t[0], rng.normal(size=4)+1j*rng.normal(size=4))
            y=('B',B_elem()) if t[1]=='B' else (t[1], rng.normal(size=4)+1j*rng.normal(size=4))
            z=('B',B_elem()) if t[2]=='B' else (t[2], rng.normal(size=4)+1j*rng.normal(size=4))
            Jp=J(x,y,z,1.,0.); Jq=J(x,y,z,0.,1.)
            for a,b in zip(np.asarray(Jp).flat,np.asarray(Jq).flat):
                rows.append([a.real,b.real]); rows.append([a.imag,b.imag])
    A=np.array(rows); u,s,vh=np.linalg.svd(A)
    ns=vh[s<1e-7*max(s.max(),1e-9)]
    return s, ns
for tr,name in [(False,'gl(4)'),(True,'sl(4) B-elements')]:
    s,ns=stats(tr)
    print(f"{name}: svals={np.round(s[:4],4)}  null-dim={len(ns)}  dirs={[np.round(v/np.linalg.norm(v),6).tolist() for v in ns[:2]]}")
# image-in-sl(4) condition separately: tr K(p,q) = (p+4q) f(v)
print("image condition tr[F,z] = (p+4q) z^T F -> sl(4) requires p+4q=0; with q=-p -> -3p=0 -> p=q=0")
