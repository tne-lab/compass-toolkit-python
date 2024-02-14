"""
Translation of MATLAB code 'compass_em.m' to python
by Sumedh Nagrale
17th Sep 2023
"""

import numpy as np
from scipy.stats import norm


#  Normal parameters - C,D and Sv
def normal_param_cdv(p, ck, dk, c_fill_ind, d_fill_ind, obs_valid, MCk, xM, Yn, XSmt, In, SSmt):
    # replace params

    sv = p[0]  # zero location
    # The logic is to start from the second point so that we have the location from 1 and so on
    ck[c_fill_ind] = p[1:1 + len(c_fill_ind)]  # removing 1 from the script
    dk[d_fill_ind] = p[1 + len(c_fill_ind):]  # the next element after the ck portion
    # sol the logic should be
    # 0 [po p1 p1 p1 p2 p2]
    # len(c_fill_ind) = 3
    # p[1:len(c_fill_ind)] = p[1:3] = p[1,2,3]
    # 1: len(c_fill_ind) basically c_fill_ind is the location
    # p[1 + len(c_fill_ind):end]= p[1+3:] -- p[4,5]
    # [po p2 p2]
    # 0
    #
    # function value
    f = np.zeros((1, 1))

    # now, calculate a part
    val_ind = np.where(obs_valid)[0]
    for l in range(len(val_ind)):
        # valid indexes
        z = val_ind[l]

        # param on time
        ctk = ck * MCk[z] @ xM
        dtk = dk

        dy = Yn[z] - (ctk @ XSmt[z] + dtk @ In[z].T)
        sy = ctk @ SSmt[z] @ ctk.T

        if obs_valid[z] == 1:
            f = f + 0.5 * np.log(sv) + (dy ** 2 + sy) / (2 * sv)

        if obs_valid[z] == 2:
            h0 = dy / np.sqrt(sv)
            # incomplete gamma at h0
            # Survival function (1 - cdf) at x of the given RV. (to avoid confusion in future), to get upper
            g0 = norm.sf(h0, loc=0, scale=1)
            # derivative of log of incomplete
            g1 = norm.pdf(h0, loc=0, scale=1) / g0
            # if np.isnan(g1):
            #     print(p, '', z)
            gt = (1 / np.sqrt(sv)) * h0 * g1 - g1 ** 2
            f = f - np.log(g0) - 0.5 * sy * gt
    return f[0]
