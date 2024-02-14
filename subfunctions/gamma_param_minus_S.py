"""
Translation of MATLAB code 'compass_em.m' to python
by Sumedh Nagrale
17th Sep 2023
"""

import numpy as np
from scipy.special import gamma, gammaincc
from scipy.stats import gamma as gammapdf


# Gamma Parameter Estimation
def gamma_param_minus_S(p, ck, dk, c_fill_ind, d_fill_ind, obs_valid, MCk, xM, Yn, XSmt, In, SSmt,S):
    # (p, Yn, S, ck, c_fill_ind, dk, d_fill_ind, obs_valid, MCk, xM, XSmt, SSmt, In):
    # first parameter is v
    # second parameter is alpha (alpha is positive and smaller than minimum Yn)
    # yk - paper EMBC
    yk = Yn - S
    # define v
    v = p[0]
    ck[c_fill_ind] = p[1:1 + len(c_fill_ind)]  # removing 1 from the script
    dk[d_fill_ind] = p[1 + len(c_fill_ind):]  # the next element after the ck portion
    # function value
    f = np.zeros((1, 1))
    # now, calculate a part
    val_ind = np.where(obs_valid)[0]
    for l in range(len(val_ind)):
        # valid indexes
        t = val_ind[l]
        # param on time
        ctk = ck * MCk[t] @ xM
        dtk = dk
        # ey/sy
        ey = ctk @ XSmt[t] + dtk @ In[t].T
        sy = ctk @ SSmt[t] @ ctk.T
        # common term
        if obs_valid[t] == 1:
            f = f + np.log(gamma(v)) - v * np.log(yk[t] * v) + v * ey + np.log(yk[t]) + v * yk[t] * np.exp(
                -ey + 0.5 * sy)
        if obs_valid[t] == 2:
            # point at mean
            h0 = v * yk[t] * np.exp(-ey)
            # incomplete gamma at h0
            g0 = gammaincc(v, h0)  # gammainc(h0, v, upper=True)
            g1 = gammapdf.pdf(h0, v, scale=1)
            # derivative of log of incomplete
            g2 = g1 / g0
            gt = (v - h0) * h0 * g2 + h0 * h0 * g2 * g2
            f = f - np.log(g0) + 0.5 * (ctk @ SSmt[t] @ ctk.T) * gt
    return f[0]
