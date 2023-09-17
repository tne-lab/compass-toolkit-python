import numpy as np


def BernoulliParam(p, e_fill_ind, f_fill_ind, ek, fk, obs_valid, MEk, xM, XSmt, SSmt, Ib, Yb):
    # replace param
    ek[e_fill_ind] = p[:len(e_fill_ind)]
    fk[f_fill_ind] = p[len(e_fill_ind):]
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
        f -= Yb[z] * ey + np.log(1 + np.exp(ey)) + 0.5 * pt * (1 - pt) * sy
    return f

'''
# Example input variables
e_fill_ind = [1, 2, 3]
f_fill_ind = [4, 5, 6]
obs_valid = np.array([1, 1, 0, 1, 0])
Yb = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
ek = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
MEk = {0: np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
       1: np.array([0.2, 0.3, 0.4, 0.5, 0.6])}
xM = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
XSmt = {0: np.array([0.1, 0.2, 0.3]),
        1: np.array([0.4, 0.5, 0.6])}
fk = np.array([0.1, 0.2, 0.3])
Ib = np.array([[0.1, 0.2, 0.3],
               [0.4, 0.5, 0.6]])
SSmt = {0: np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]),
        1: np.array([[9, 8, 7],
                    [6, 5, 4],
                    [3, 2, 1]])}
'''