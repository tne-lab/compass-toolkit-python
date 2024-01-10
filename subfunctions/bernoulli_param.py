"""
Translation of MATLAB code 'compass_em.m' to python
by Sumedh Nagrale
17th Sep 2023
"""

import numpy as np


def bernoulli_param(p, e_fill_ind, f_fill_ind, ek, fk, obs_valid, MEk, xM, XSmt, SSmt, Ib, Yb):
    # replace param
    ek[e_fill_ind] = p[:len(e_fill_ind)]
    fk[f_fill_ind] = p[len(e_fill_ind):]  # no need to add 1
    # function calculation
    f = 0
    # now, calculate a part
    val_ind = np.where(obs_valid == 1)[0]
    for l in range(len(val_ind)):
        # valid indexes
        z = val_ind[l]
        # param on time
        etk = np.dot(ek * MEk[z], xM)
        ftk = fk
        ey = np.dot(etk, XSmt[z]) + np.dot(ftk, Ib[z])
        sy = np.dot(etk, np.dot(SSmt[z], etk.T))
        pt = 1 / (1 + np.exp(-ey))
        f = f - Yb[z] * ey + np.log(1 + np.exp(ey)) + 0.5 * pt * (1 - pt) * sy
    return f[0]
