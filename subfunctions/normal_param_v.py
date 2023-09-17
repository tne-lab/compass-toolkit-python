"""
Translation of MATLAB code 'compass_em.m' to python
by Sumedh Nagrale
17th Sep 2023
"""

import numpy as np
from scipy.stats import norm


# Normal parameters - C,D
def NormalParamV(p, Ck, Dk, obs_valid, MCk, xM, Yn, XSmt, In, SSmt):
    # replace params
    ck = Ck
    dk = Dk
    sv = p[0]
    # function value
    f = 0
    # now, calculate a part
    val_ind = np.where(obs_valid)[0]
    for l in range(len(val_ind)):
        # valid indexes
        z = val_ind[l]
        # param on time
        ctk = (ck * MCk[z]) @ xM
        dtk = dk
        
        dy = Yn[z] - ctk @ XSmt[z] - dtk @ In[z].T
        sy = ctk @ SSmt[z] @ ctk.T
        if obs_valid[z] == 1:
            f += 0.5 * np.log(sv) + (dy ** 2 + sy) / (2 * sv)

        if obs_valid[z] == 2:
            h0 = dy / np.sqrt(sv)
            # incomplete gamma at h0
            # Survival function (1 - cdf) at x of the given RV. (to avoid confusion in future), to get upper
            g0 = norm.sf(h0, loc=0, scale=1)
            # derivative of log of incomplete
            g1 = norm.pdf(h0, loc=0, scale=1) / g0
            gt = (1 / np.sqrt(sv)) * h0 * g1 - g1 ** 2
            f -= np.log(g0) - 0.5 * sy * gt
    return f
