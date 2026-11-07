import numpy as np
def combat(data, batch, covars=None, eb=True):
    """Parametric empirical-Bayes ComBat (Johnson et al. 2007). data: samples x features.
    batch: array of labels; covars: samples x q array of covariates to preserve. Returns harmonized samples x features."""
    Y=np.asarray(data,float).T            # features x samples
    p,n=Y.shape
    batch=np.asarray(batch); levels=list(dict.fromkeys(batch)); nb=len(levels)
    D=np.zeros((n,nb))
    for j,b in enumerate(levels): D[batch==b,j]=1
    parts=[D]
    if covars is not None:
        C=np.asarray(covars,float).reshape(n,-1); parts.append(C)
    design=np.hstack(parts)
    # regression
    beta=np.linalg.lstsq(design,Y.T,rcond=None)[0]          # (nb+q) x p
    n_b=D.sum(0)
    grand=(n_b/n)@beta[:nb]                                  # p
    resid=Y.T-design@beta
    var_pooled=(resid**2).mean(0)                            # p
    stand_mean=np.tile(grand,(n,1))
    if covars is not None:
        stand_mean=stand_mean+C@beta[nb:]
    s=(Y.T-stand_mean)/np.sqrt(var_pooled)                   # n x p
    # batch effects
    gamma_hat=np.linalg.lstsq(D,s,rcond=None)[0]             # nb x p
    delta_hat=np.array([s[batch==b].var(0,ddof=1) for b in levels])
    if not eb:
        gs,ds=gamma_hat,delta_hat
    else:
        gbar=gamma_hat.mean(1); t2=gamma_hat.var(1,ddof=1)
        def aprior(d): m=d.mean(); s2=d.var(ddof=1); return (2*s2+m**2)/s2
        def bprior(d): m=d.mean(); s2=d.var(ddof=1); return (m*s2+m**3)/s2
        gs=np.zeros_like(gamma_hat); ds=np.zeros_like(delta_hat)
        for i,b in enumerate(levels):
            idx=batch==b; x=s[idx]; nI=idx.sum()
            a=aprior(delta_hat[i]); bb=bprior(delta_hat[i])
            g=gamma_hat[i].copy(); d=delta_hat[i].copy()
            for _ in range(200):
                g_new=(nI*t2[i]*gamma_hat[i]+d*gbar[i])/(nI*t2[i]+d)
                d_new=(0.5*((x-g_new)**2).sum(0)+bb)/(nI/2.0+a-1.0)
                ch=max(np.abs(g_new-g).max(),np.abs(d_new-d).max()); g,d=g_new,d_new
                if ch<1e-6: break
            gs[i],ds[i]=g,d
    out=np.zeros_like(s)
    for i,b in enumerate(levels):
        idx=batch==b
        out[idx]=(s[idx]-gs[i])/np.sqrt(ds[i])
    out=out*np.sqrt(var_pooled)+stand_mean
    return out
