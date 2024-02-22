"""
Translation of MATLAB code 'compass_deviance.m' to python
16 Sep 2023  Translated by Sumedh Nagrale
"""
import numpy as np
import scipy.stats as stats

import compass_toolkit.compass_Tk as Ctk
import compass_toolkit.compass_Qk as Cqk
import compass_toolkit.compass_post_sampling as cps

def compass_deviance(DISTR=None, In=None, Ib=None, Yn=None, Yb=None, Param=None, obs_valid=None, XSmt=None, SSmt=None):
    # This function calculates the deviance for both continuous and discrete observations
    # number of samples over censored points
    Ns = 10
    K = np.maximum(len(Yn), len(Yb))
    ''' Here, you need the function to replace the censored/missing data with a sampleed one '''
    if np.any(Yn):
        tYn = np.tile(Yn, (1, Ns))

    if np.any(Yb):
        tYb = np.tile(Yb, (1, Ns))

    for k in range(0, K):
        if obs_valid[k] == 0:  # missing
            for s in range(0, Ns):
                # function call
                [samp_yn, samp_yb] = cps.compass_post_sampling(DISTR, None, In[k, :].reshape(-1, 1),
                                                               Ib[k, :].reshape(-1, 1), Param, XSmt[k], SSmt[k])

                if DISTR[0]:
                    tYn[k, s] = samp_yn
                if DISTR[1]:
                    tYb[k, s] = samp_yb

        if obs_valid[k] == 2:  # censored
            for s in range(1, Ns):
                # function call
                samp_yn, samp_yb = cps.compass_post_sampling(DISTR, Param['censor_time'], In[k, :].reshape(1, -1),
                                                             Ib[k, :].reshape(1, -1), Param,
                                                             XSmt[k], SSmt[k])
                if DISTR[0]:
                    tYn[k][s] = samp_yn
                if DISTR[1]:
                    tYb[k][s] = samp_yb

    '''CONTINUOUS PART DEVIANCE'''
    '''Normal'''
    DEV_C = []
    if DISTR[0] == 1:
        xM = Param['xM']
        [MCk, MDk] = Ctk.compass_Tk(In, Param)
        Vk = Param['Vk']
        DTk = Param['Dk'] * MDk
        Ck = Param['Ck']
        # deviance calculation
        N = 1000
        DEV_C = 0
        for k in range(0, K):
            # draw samples
            Xs = np.random.multivariate_normal(mean=XSmt[k][0],cov= SSmt[k], size=N).T
            CTk = (Ck * MCk[k]) @ xM
            Mx = CTk @ Xs + DTk @ In[k].T
            Sx = Vk
            if obs_valid[k] == 1:
                DEV_C = DEV_C - 2 * np.sum(np.log(stats.norm.pdf(tYn[k, 0], Mx, np.sqrt(Sx)))) / N
            if obs_valid[k] == 2:
                avg_log_ll = 0
                for s in range(1, Ns):
                    avg_log_ll = avg_log_ll - 2 * np.sum(np.log(stats.norm.pdf(tYn[k, s], Mx, np.sqrt(Sx)))) / N
                avg_log_ll = avg_log_ll / Ns
                DEV_C = DEV_C + avg_log_ll

    '''Gamma'''
    if DISTR[0] == 2:
        xM = Param['xM']
        [MCk, MDk] = Ctk.compass_Tk(In, Param)
        DTk = Param['Dk'] * MDk
        '''model parameters'''
        Ck = Param['Ck']
        Vk = Param['Vk']
        S = Param['S']
        '''draw samples per trial'''
        N = 1000
        DEV_C = 0
        for k in range(0, K):
            # draw samples
            Xs = np.random.multivariate_normal(mean=XSmt[k][0], cov= SSmt[k], size=N).T
            CTk = (Ck * MCk[k]) @ xM
            Mx = np.exp(CTk @ Xs + DTk @ In[k].T)
            Sx = Vk
            if obs_valid[k] == 1:
                DEV_C = DEV_C - 2 * np.sum(np.log(stats.gamma.pdf(tYn[k, 0] - S, a=Vk, scale=Mx / Vk))) / N

            if obs_valid[k] == 2:
                avg_log_ll = 0
                for s in range(1, Ns):
                    avg_log_ll = avg_log_ll - 2 * np.sum(np.log(stats.gamma.pdf(x=tYn[k, s] - S, a=Vk, scale=Mx / Vk))) / N
                avg_log_ll = avg_log_ll / Ns
                DEV_C = DEV_C + avg_log_ll

    '''Discrete Part DEVIANCE'''
    '''we assume fully observed data'''
    DEV_D = []
    if DISTR[1] == 1:
        xM = Param['xM']
        [MEk, MFk] = Cqk.compass_Qk(In, Param)
        '''model parameters'''
        Ek = Param['Ek']
        FTk = Param['Fk'] * MFk
        '''draw samples per trial'''
        N = 1000
        '''map to larger space'''
        DEV_D = 0
        for k in range(0, K):
            Xs = np.random.multivariate_normal(mean=XSmt[k][0], cov=SSmt[k], size=N).T
            ETk = (Ek * MEk[k]) @ xM
            st = ETk @ Xs + FTk @ Ib[k, :].T
            pk = np.divide(np.exp(st), (1 + np.exp(st)))
            if obs_valid[k] == 1:
                if tYb[k, 0]:
                    DEV_D = DEV_D - 2 * np.sum(np.log(pk)) / N
                else:
                    DEV_D = DEV_D - 2 * np.sum(np.log(1 - pk)) / N
            if obs_valid[k] == 2:
                avg_log_ll = 0
                for s in range(0, Ns):
                    if tYb[k, s]:
                        avg_log_ll = avg_log_ll - 2 * np.sum(np.log(pk)) / N
                    else:
                        avg_log_ll = avg_log_ll - 2 * np.sum(np.log(1 - pk)) / N
                avg_log_ll = avg_log_ll / Ns
                DEV_D = DEV_D + avg_log_ll

    if DISTR[1] == 1:
        return [DEV_C, DEV_D]
    else:
        return [DEV_C, None]


'''
print(compass_deviance(DISTR=[2, 0],
                       In=np.array([[1, 0, 1]]),
                       obs_valid=np.array([0, 0, 0]),
                       Ib=np.array([[1, 0, 1]]),
                       Yn=np.array([[0.8, 1, 0.68]]),
                       Yb=np.array([[0.8, 1, 0.68]])))
'''
