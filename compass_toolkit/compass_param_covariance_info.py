import numpy as np

from scipy.special import polygamma

import compass_toolkit.compass_post_sampling as cps
import compass_toolkit.compass_Tk as Ctk
import compass_toolkit.compass_Qk as Cqk


# line 21-24 and line 28-31
# line 34 is slow,
# Change all the dictionary calls to variables use
def compass_param_covariance_info(DISTR, Uk, In, Ib, Yn, Yb, Param, obs_valid, XSmt, SSmt):
    K = max(len(Yn), len(Yb))

    COV_X = []
    COV_C = {}
    COV_D = {}

    for k in range(1, K + 1):
        if obs_valid[k - 1] == 0:  # missing
            samp_yn, samp_yb = cps.compass_post_sampling(DISTR, None, In[k - 1], Ib[k - 1], Param, XSmt[k - 1],
                                                         SSmt[k - 1])
            if DISTR[0]:
                Yn[k - 1] = samp_yn
            if DISTR[1]:
                Yb[k - 1] = samp_yb
        if obs_valid[k - 1] == 2:  # censored
            samp_yn, samp_yb = cps.compass_post_sampling(DISTR, Param['censor_time'], In[k - 1].reshape(1, -1),
                                                         Ib[k - 1].reshape(1, -1), Param, XSmt[k - 1], SSmt[k - 1])
            if DISTR[0]:
                Yn[k - 1] = samp_yn
            if DISTR[1]:
                Yb[k - 1] = samp_yb

    for r in range(len(XSmt[0])):
        temp = Param['X0'] @ Param['X0'].T + Param['W0']
        for k in range(1, K):
            temp += XSmt[k - 1] @ XSmt[k - 1].T + SSmt[k - 1]

        tempA = temp

        if Uk.shape[0] > 0:
            temp = np.zeros((Uk.shape[1], Uk.shape[1]))
            for k in range(1, K):
                temp += np.outer(Uk[k - 1], Uk[k - 1])

            tempB = temp
            temp = np.zeros((len(XSmt[0]), Uk.shape[1]))
            for k in range(1, K):
                temp += XSmt[k - 1] @ np.atleast_2d(Uk[k - 1])

            tempAB = temp
        else:
            tempB = None
            tempAB = None

        temp = Param['Wk'][r, r] * np.linalg.pinv(np.block([[tempA, tempAB], [tempAB.T, tempB]]))

        COV_X_r = {}
        COV_X_r['A'] = temp[:Param['Ak'].shape[0], :Param['Ak'].shape[0]]
        COV_X_r['AB'] = temp[:Param['Ak'].shape[0], Param['Ak'].shape[0]:]
        COV_X_r['B'] = temp[Param['Ak'].shape[0]:, Param['Ak'].shape[0]:]
        COV_X_r['W'] = Param['Wk'][r, r] ** 2 * 2 / K
        COV_X_r['SE_W'] = np.sqrt(np.diag([COV_X_r['W']]))

        COV_X.append(COV_X_r)

    xM = Param['xM']
    COV_C = {}
    if DISTR[0] == 1:
        MCk, MDk = Ctk.compass_Tk(In, Param)
        tXSmt = xM @ XSmt[0]
        tSSmt = xM @ SSmt[0] @ xM.T
        tempA = np.outer(MCk[0], MCk[0]) * (tXSmt @ tXSmt.T + tSSmt)
        tempB = np.outer(MDk, MDk) * (In[0][:, np.newaxis] @ In[0][np.newaxis, :])
        tempAB = np.outer(MCk[0], tXSmt) * (MDk * In[0])

        for k in range(1, K):
            tXSmt = xM @ XSmt[k]
            tSSmt = xM @ SSmt[k] @ xM.T

            tempA += np.outer(MCk[k], MCk[k]) * (tXSmt @ tXSmt.T + tSSmt)
            tempB += np.outer(MDk, MDk) * (In[k][:, np.newaxis] @ In[k][np.newaxis, :])
            tempAB += np.outer(MCk[k], tXSmt) * (MDk * In[k])

        temp = np.block([[tempA, tempAB], [tempAB.T, tempB]])
        ind = np.concatenate((np.arange(tempA.shape[0]), np.arange(tempA.shape[0]) + len(MDk)))
        temp = temp[:, ind]
        temp = temp[ind, :]
        temp_x = np.linalg.pinv(temp) * Param['Vk']
        temp = np.zeros((len(MCk[0]) + len(MDk), len(MCk[0]) + len(MDk)))
        temp[np.ix_(ind, ind)] = temp_x

        COV_C['C'] = temp[:len(Param['Ck']), :len(Param['Ck'])]
        COV_C['SE_C'] = np.sqrt(np.diag(COV_C['C']))
        COV_C['CD'] = temp[:len(Param['Ck']), len(Param['Ck']):]
        COV_C['D'] = temp[len(Param['Ck']):, len(Param['Ck']):]
        COV_C['SE_D'] = np.sqrt(np.diag(COV_C['D']))
        COV_C['V'] = Param['Vk'] ** 2 * 2 / K
        COV_C['SE_V'] = np.sqrt(COV_C['V'])

    if DISTR[0] == 2:
        MCk, MDk = Ctk.compass_Tk(In, Param)
        Ck = Param['Ck']
        Dk = Param['Dk']
        Vk = Param['Vk']
        S = Param['S']
        N = 1000
        Xs = np.dot(xM, np.random.multivariate_normal(XSmt[0][0], SSmt[0], N).T)
        temp = (Ck * MCk[0]) @ Xs + (Dk * MDk) @ In[0]
        Mx = np.exp(-temp)
        Y = Yn[0] - S
        Xs_a = np.tile(Mx, (len(Ck), 1)) * Xs
        tempCC = (-Vk * Y * (MCk[0].T @ MCk[0]) @ (Xs_a @ Xs.T)) / N
        tempCD = -Vk * Y * (MCk[0] @ np.sum(Xs_a, axis=1).T / N) * (MDk * In[0])
        tempDD = -Vk * Y * (np.dot(MDk.T, MDk) * (np.dot(In[0][:, np.newaxis], In[0][np.newaxis, :])) * np.sum(Mx)) / N
        tempCV = -np.dot(MCk[0], np.sum(Xs - Y * Xs_a, axis=1)[:, np.newaxis]) / N
        tempCS = -Vk * np.dot(MCk[0], np.sum(Xs_a, axis=1)[:, np.newaxis]) / N
        tempDV = -MDk * (1 - Y * np.sum(Mx) / N)
        tempDS = -Vk * np.dot(MDk, np.sum(Mx)) / N
        tempSS = (1 - Vk) / Y ** 2
        tempVV = -polygamma(1, Vk) + 1 / Vk
        tempSV = (-1 / Y) + np.sum(Mx) / N

        for k in range(1, K):
            Xs = np.dot(xM, np.random.multivariate_normal(XSmt[k][0], SSmt[k], N).T)
            temp = (Ck * MCk[k]) @ Xs + (Dk * MDk) @ In[k]
            Mx = np.exp(-temp)
            Y = Yn[k] - S
            Xs_a = np.tile(Mx, (len(Ck), 1)) * Xs
            tempCC = tempCC - (Vk * Y * (MCk[k].T @ MCk[k]) @ (Xs_a @ Xs.T)) / N
            tempDD -= Vk * Y * (
                    np.dot(MDk.T, MDk) * (np.dot(In[k][:, np.newaxis], In[k][np.newaxis, :])) * np.sum(Mx)) / N
            tempCD = tempCD - (Vk * Y * (MCk[k] @ np.sum(Xs_a, axis=1).T / N) * (MDk * In[k]))
            tempCV -= np.dot(MCk[k], np.sum(Xs - Y * Xs_a, axis=1)[:, np.newaxis]) / N
            tempCS -= Vk * np.dot(MCk[k], np.sum(Xs_a, axis=1)[:, np.newaxis]) / N
            tempDV = tempDV - MDk * (1 - Y * np.sum(Mx) / N)  # np.dot(MDk, 1 - Y * np.sum(Mx) / N)
            tempDS -= Vk * np.dot(MDk, np.sum(Mx)) / N
            tempSS += (1 - Vk) / Y ** 2
            tempVV = tempVV - polygamma(1, Vk) + 1 / Vk
            tempSV += (-1 / Y) + np.sum(Mx) / N

        temp = np.block([[tempCC, tempCD],
                         [tempCD.T, tempDD]])
        n_col = np.hstack((tempCV, tempDV)).T
        temp = np.hstack((temp, n_col))
        n_col = np.hstack((tempCS, tempDS)).T
        temp = np.hstack((temp, n_col))
        n_row = np.hstack((tempCV, tempDV, tempVV, [tempSV]))
        temp = np.vstack((temp, n_row))
        n_row = np.hstack((tempCS, tempDS, [tempSV], tempSS))
        temp = np.vstack((temp, n_row))

        if S != 0:
            ind = np.concatenate((np.arange(tempCC.shape[0]), np.arange(tempCC.shape[0]) + len(MDk),
                                  np.arange(tempCC.shape[0]) + len(MDk.T) + 1,
                                  np.arange(tempCC.shape[0]) + len(MDk.T) + 2))
            temp = temp[:, ind]
            temp = temp[ind, :]
            temp_x = -np.linalg.pinv(temp)
            temp = np.zeros((len(MCk[0]) + len(MDk.T) + 2, len(MCk[0]) + len(MDk.T) + 2))
            temp[np.ix_(ind, ind)] = temp_x
        else:
            ind = np.concatenate((np.arange(tempCC.shape[0]), np.arange(tempCC.shape[0]) + len(MDk.T),
                                  np.arange(tempCC.shape[0]) + len(MDk.T) + 1))
            temp = temp[:, ind]
            temp = temp[ind, :]
            temp_x = -np.linalg.pinv(temp)
            temp = np.zeros((len(MCk[0]) + len(MDk) + 2, len(MCk[0]) + len(MDk) + 2))
            temp[np.ix_(ind, ind)] = temp_x

        COV_C['C'] = temp[:len(Param['Ck']), :len(Param['Ck'])]
        COV_C['SE_C'] = np.sqrt(np.diag(COV_C['C']))
        COV_C['D'] = temp[len(Param['Ck']):len(Param['Ck']) + len(Param['Dk'].T),
                     len(Param['Ck']):len(Param['Ck']) + len(Param['Dk'].T)]
        COV_C['SE_D'] = np.sqrt(np.diag(COV_C['D']))
        COV_C['S'] = temp[-1, -1]
        COV_C['Vk'] = temp[-2, -2]

        COV_C['CD'] = temp[:len(Param['Ck']), len(Param['Ck']):len(Param['Ck']) + len(Param['Dk'].T)]
        COV_C['CV'] = temp[:len(Param['Ck']), len(Param['Ck']) + len(Param['Dk'].T)]
        COV_C['CS'] = temp[:len(Param['Ck']), len(Param['Ck']) + len(Param['Dk'].T) + 1]
        COV_C['DV'] = temp[len(Param['Ck']):len(Param['Ck']) + len(Param['Dk'].T),
                      len(Param['Ck']) + len(Param['Dk'].T)]
        COV_C['DS'] = temp[len(Param['Ck']):len(Param['Ck']) + len(Param['Dk'].T),
                      len(Param['Ck']) + len(Param['Dk'].T) + 1]
        COV_C['VS'] = temp[-1, -2]

    if DISTR[1] == 1:
        MEk, MFk = Cqk.compass_Qk(Ib, Param)
        Ek = Param['Ek']
        Fk = Param['Fk']
        N = 1000
        Xs = np.dot(xM, np.random.multivariate_normal(XSmt[0][0], SSmt[0], N).T)
        temp = (Ek * MEk[0]) @ Xs + (Fk * MFk) @ Ib[0]
        Ps = np.exp(temp) / (1 + np.exp(temp))
        Xs_a = np.tile(Ps * (1 - Ps), (len(Ek), 1)) * Xs
        Xs_b = Xs
        tempA = (MEk[0].T @ MEk[0]) * (Xs_a @ Xs_b.T) / N
        tempB = (MFk.T @ MFk) * (In[0][:, np.newaxis] @ In[0][np.newaxis, :]) * np.sum(Ps * (1 - Ps)) / N
        tempAB = (MEk[0] * np.sum(Xs_a, axis=1).T / N).T @ (MFk * In[0])

        for k in range(1, K):
            Xs = np.dot(xM, np.random.multivariate_normal(XSmt[k][0], SSmt[k], N).T)
            temp = (Ek * MEk[k]) @ Xs + (Fk * MFk) @ Ib[k]
            Ps = np.exp(temp) / (1 + np.exp(temp))
            Xs_a = np.tile(Ps * (1 - Ps), (len(Ek), 1)) * Xs
            Xs_b = Xs
            tempA += (MEk[k].T @ MEk[k]) * (Xs_a @ Xs_b.T) / N
            tempB += (MFk.T @ MFk) * (In[k][:, np.newaxis] @ In[k][np.newaxis, :]) * np.sum(Ps * (1 - Ps)) / N
            tempAB += (MEk[k] * np.sum(Xs_a, axis=1).T / N).T @ (MFk * In[k])

        temp = np.block([[tempA, tempAB],
                         [tempAB.T, tempB]])
        ind = np.concatenate((np.arange(tempA.shape[0]), np.arange(tempA.shape[0]) + 1 + np.where(MFk != 0)[0]))
        temp = temp[:, ind]
        temp = temp[ind, :]
        temp_x = np.linalg.pinv(temp)
        temp = np.zeros((len(MEk[0]) + len(MFk.T), len(MEk[0]) + len(MFk.T)))
        temp[np.ix_(ind, ind)] = temp_x

        COV_D['E'] = temp[:len(Param['Ek']), :len(Param['Ek'])]
        COV_D['SE_E'] = np.sqrt(np.diag(COV_D['E']))
        COV_D['EF'] = temp[:len(Param['Ek']), len(Param['Ek']):]
        COV_D['F'] = temp[len(Param['Ek']):, len(Param['Ek']):]
        COV_D['SE_F'] = np.sqrt(np.diag(COV_D['F']))

    return COV_X, COV_C, COV_D
