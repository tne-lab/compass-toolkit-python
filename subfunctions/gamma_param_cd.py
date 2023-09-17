"""
Translation of MATLAB code 'compass_em.m' to python
by Sumedh Nagrale
17th Sep 2023
"""

import numpy as np
from scipy.special import gamma, gammainc


def GammaParamCD(p, Yn, S, Vk, ck, c_fill_ind, dk, d_fill_ind, obs_valid, MCk, xM, XSmt, SSmt, In):
    # first parameter is v
    # second parameter is alpha (alpha is positive and smaller than minimum Yn)
    # yk - paper EMBC
    yk = Yn - S
    # define v
    v = Vk
    # ctk
    ck[c_fill_ind] = p[0:len(c_fill_ind)]
    # dtk - parameters linked to Input
    # adding 1 to the index
    dk[d_fill_ind] = p[1+len(c_fill_ind):]
    # function value
    f = 0
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
            f += np.log(gamma(v)) - v * np.log(yk[t] * v) + v * ey + np.log(yk[t]) + v * yk[t] * np.exp(
                -ey + 0.5 * sy)
        if obs_valid[t] == 2:
            # point at mean
            h0 = v * yk[t] * np.exp(-ey)
            # incomplete gamma at h0
            g0 = gammainc(h0, v, 'upper')
            g1 = gamma.pdf(h0, v, 0, 1)
            # derivative of log of incomplete
            g2 = g1 / g0
            gt = (v - h0) * h0 * g2 + h0 * h0 * g2 * g2
            f -= np.log(g0) + 0.5 * (ctk @ SSmt[t] @ ctk.T) * gt
    return f
