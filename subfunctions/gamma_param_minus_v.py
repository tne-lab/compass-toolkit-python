"""
Translation of MATLAB code 'compass_em.m' to python
by Sumedh Nagrale
17th Sep 2023
"""

import numpy as np
from scipy.special import gamma, gammaincc


# Gamma Paramater Estimation
def gamma_param_minus_v(p, Yn, Vk, ck, dk, MCk, xM, In, XSmt, SSmt, c_fill_ind, d_fill_ind, obs_valid):
    # first parameter is v
    # second parameter is alpha (alpha is positive and smaller than minimum Yn)
    # yk - paper EMBC
    yk = Yn - p[0]
    # define v
    v = Vk
    # ctk
    ck[c_fill_ind] = p[1:0 + len(c_fill_ind)]
    # dtk - parameters linked to Input
    dk[d_fill_ind] = p[1 + len(c_fill_ind):]
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
            h0 = v * yk[t] / np.exp(-ey)
            # incomplete gamma at h0
            g0 = gamma(v) * gammaincc(v, h0)
            # derivative of log of incomplete
            g1 = -((h0 ** (v - 1)) * np.exp(-h0)) / g0
            g2 = g1 * (((v - 1) / h0) - 1) - g1 ** 2
            gt = g2 * h0 * h0 + g1 * h0
            f = f + np.log(gamma(v)) - np.log(g0) - 0.5 * (ctk @ SSmt[t] @ ctk.T) * gt
    return f[0]
