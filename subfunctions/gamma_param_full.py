"""
Translation of MATLAB code 'compass_em.m' to python
by Sumedh Nagrale
17th Sep 2023
"""

import numpy as np
from scipy.special import gamma, gammainc


# Gamma Parameter Estimation
def gamma_param_full(p, Yn, obs_valid, ck, dk, MCk, xM, XSmt, SSmt, In, c_fill_ind, d_fill_ind):
    # first parameter is v
    # second parameter is alpha (alpha is positive and smaller than minimum Yn)
    # yk - paper EMBC
    yk = Yn - p[1]
    # define v
    v = p[0]
    # ctk
    ck[c_fill_ind] = p[2:1 + len(c_fill_ind)]
    # dtk - parameters linked to Input
    dk[d_fill_ind] = p[2 + len(c_fill_ind):]

    # function value
    f = 0
    # now, calculate a part
    val_ind = np.where(obs_valid)[0]
    for l in range(len(val_ind)):
        # valid indexes
        t = val_ind[l]
        # param on time
        ctk = (ck * MCk[t]) @ xM
        dtk = dk
        # ey/sy
        ey = ctk @ XSmt[t] + dtk @ (In[t]).T
        sy = ctk @ SSmt[t] @ ctk.T
        # common term
        if obs_valid[t] == 1:
            f += np.log(gamma(v)) - v * np.log(yk[t] * v) + v * ey + np.log(yk[t]) + v * yk[t] * np.exp(
                -ey + 0.5 * sy)
        if obs_valid[t] == 2:
            # point at mean
            h0 = v * yk[t] / np.exp(ey)
            # incomplete gamma at h0
            g0 = gamma(v) * gammainc(h0, v, upper=True)
            # derivative of log of incomplete
            g1 = -((h0 ** (v - 1)) * np.exp(-h0)) / g0
            g2 = -g1 ** 2 + g1 * (((v - 1) / h0) - 1)
            gt = g2 * h0 * h0 + g1 * h0
            f += np.log(gamma(v)) - np.log(g0) - 0.5 * ctk @ SSmt[t] @ ctk * gt
    return f