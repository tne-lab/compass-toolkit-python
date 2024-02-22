
import numpy as np
from scipy.stats import norm, gamma
from scipy.special import gammaincc

import subfunctions as objective_function
import compass_toolkit as compass

def compass_em(DISTR=None, Uk=None, In=None, Ib=None, Yn=None, Yb=None, Param=None, obs_valid=None):
    global MFk, K, Ck, Dk, Vk, censor_time, Ek, Fk, MEk, pk, S, ck, dk, Yp
    ####################################Setup part################################################
    ''' Input Argument
         DISTR, a vector of two variables. The [1 0] means there is only normal
         observation/s, [0 1] means there is only binary observation/s, and [1 1]
         will be both observations.
         Uk: is a matrix of size KxS1 - K is the length of observation - input to
         State model - X(k)=A*X(k-1)+B*Uk+Wk
         In: is a matrix of size KxS3 - K is the length of observation - input to Normal observation model
         Yn(k)=(C.*MCk)*X(k)+(D.*MDk)*In+Vk       - C and D are free parameters,
         and MCk and MDk are input dependent components
         Ib: is a matrix of size KxS5 - K is the length of observation - input to Binary observation model
          P(Yb(k)==1)=sigmoid((E.*MEk)*X(k)+(F.*MFk)*Ib       - E and F are free parameters,
         and MEk and MFk are input dependent components
         Yn: is a matrix of size KxN  - K is the length of observation, matrix of
         normal observation
         Yb: is a matrix of size KxN  - K is the length of observation, matrix of
         binary observation
         Param: it keeps the model information, and paramaters
     Output Argument
         XSmt is the smoothing result - mean
         SSmt is the smoothing result - variance
         Param is the updated model parameters
         XPos is the filtering result - mean
         SPos is the filtering result - variance
         ML is the value of E-step maximization
         EYn is the prediction of the Yn
         EYb is not added yet, but it can be the prediction of binary probability'''

    '''obs_valid has three values (0= MAR, 1=observed, 2= censored)'''
    EPS = np.finfo(np.float32).tiny
    MAX_EXP = 50
    update_mode = np.copy(Param['UpdateMode'])
    '''Observation Mode, from 1 to 5'''
    if DISTR[0] == 1:
        observe_mode = DISTR[0] + 2 * DISTR[1]
    elif DISTR[0] == 2:
        observe_mode = 2 * DISTR[0] + DISTR[1]
    else:
        observe_mode = 2 * DISTR[1]

    '''Build Mask Ck, Dk ,EK and Fk - note that Ck, Ek are time dependent and the Dk and Fk is linked to a subset of 
    Input '''
    [MCk, MDk] = compass.compass_Tk(In, Param)
    if DISTR[1] == 1:
        [MEk, MFk] = compass.compass_Qk(Ib, Param)

    ''' State Space Model (X(k+1)= Ak*X(k) + Bk*Uk + Wk*iid white noise )
     ------------------
     X(k) is the state, and Uk is the input
     Ak, Bk, Wk are model paramateres
     ------------------
     Ak, MxM matrix  (M is the length of the X)'''
    Ak = np.copy(Param['Ak'])
    ''' Bk, MxS1 matrix (S1 is the length of Uk, Uk is a vector of size S1x1)'''
    Bk = np.copy(Param['Bk'])
    '''Wk, is MxS2 matrix (S2 is the length of Noise, we normally set the noise with the same dimension as the X - 
    S2=M) '''
    Wk = np.copy(Param['Wk'])
    '''X0, is a Mx1 matrix (initial value of X0)'''
    X0 = np.copy(Param['X0'])
    W0 = np.copy(Param['W0'])
    ''' This is extending x'''
    xM = np.copy(Param['xM'])

    '''Censored Reaction Time'''
    if 2 in obs_valid:
        censor_time = np.copy(Param['censor_time'])
        Yn[obs_valid != 1] = censor_time

    '''Normal/Gamma Observation Model'''
    if DISTR[0] > 0:
        '''
         For Normal,  Y(k)=(Ck.*Tk)*X(k)+Dk*Ik + Vk    Vk variance of iid white noise
         For Gamma,   Y(k)=exp((Ck.*Tk)*X(k)+Dk*Ik)    Vk is dispersion term
         ------------------
         Y(k) is the observation, and Ik is the input either indicator or continuous
         Ck, Dk, Vk are the model parameters
         Tk is model specific function - it is original set to but a one matrix
         ------------------
         Ck, 1xM matrix - (Y is an scalar observation at each time point ... - The Tk has the same size of input,
         and it is specfically designed for our task. It can be set to all 1 matrix)
         '''
        Ck = np.copy(Param['Ck'])
        ''' Bk, NxS3 matrix - (We have an input of the length S3, and Dk will be size of NxS3)'''
        Dk = Param['Dk'] * MDk
        ''' Vk, is scaler representing noise in Normal or Dispresion Term in Gamma'''
        Vk = np.copy(Param["Vk"])
        ''' Length of data '''
        K = len(Yn)
    '''Binary Observation Model (P(k)=sigmoid((Ek.*Qk)*X(k)+Fk*Ik) )'''
    if DISTR[1] == 1:
        ''' ------------------
         P(k) is the observation probability at time k, and Ik is the input either indicator or continuous
         Ek, and Fk are the model parameters
         Qk is model specific function - it is original set to but a one matrix
         ------------------'''
        ''' Ck, NxM matrix - similar to Ck, Tk'''
        Ek = np.copy(Param['Ek'])
        ''' Fk, NxS5 matrix - Similar to Dk'''
        Fk = Param['Fk'] * MFk
        '''Length of data'''
        K = len(Yb)

    ''' Gamma, Extra Parameter - Time Shift'''
    if DISTR[0] == 2:
        S = Param['S'] * 1

    '''Check Uk needs to check if the size is 0 or it's empty here'''
    if Uk.size == 0:
        Uk = np.zeros((K, Bk.shape[1]))

    '''EM Update Loop'''
    ''' Model Prediction'''
    EYn = []
    EYb = []
    if DISTR[0]:
        EYn = np.zeros((K, 1))
    if DISTR[1]:
        EYb = np.zeros((K, 1))

    ''' ML array '''
    # ML = np.array(Param['Iter']) creates numpy array of single element
    ML = {}  # To get data each iteration #[0] * Param['Iter']
    # ML = np.zeros(Param['Iter'])
    ####################################EM loop################################################
    ''' Main Loop '''
    for iter in range(1, Param['Iter'] + 1):
        ML[iter] = {}
        # display iter
        # print(f"iteration {iter} out of {Param['Iter']}") python 3.6 and above
        print('iteration ', iter, ' out of ', Param['Iter'])
        '''Run the Filtering Part'''
        '''One step prediction mean'''
        XPre = [np.array([[0]]) for _ in range(K)]  # [0] * K
        '''One step prediction covariance'''
        SPre = [np.array([[0]]) for _ in range(K)]  # [0] * K
        '''Filter mean'''
        XPos = [np.array([[0]]) for _ in range(K)]  # [[0]] * K
        '''Filter covariance'''
        SPos = [np.array([[0]]) for _ in range(K)]  # [[0]] * K
        # Filter
        for k in range(0, K):
            # One-step prediction
            if k == 0:
                XPre[k] = Ak @ X0 + Bk @ Uk[k, :].T
                SPre[k] = Ak @ W0 @ Ak.T + Wk
            else:
                XPre[k] = Ak @ XPos[k - 1] + Bk @ Uk[k, :].T
                SPre[k] = Ak @ SPos[k - 1] @ Ak.T + Wk
            # Check if the data point is censored or not
            if obs_valid[k]:
                # Draw a sample if it is censored in censor_mode 1 (sampling)
                if obs_valid[k] == 2 and Param['censor_mode'] == 1:
                    tIn, tIb, tUk = [], [], []
                    if DISTR[0]:
                        tIn = In[k, :].reshape((1, -1))
                    if DISTR[1]:
                        tIb = Ib[k, :].reshape((1, -1))
                    if Uk.size != 0:
                        tUk = Uk[k, :]
                    tYP, tYB = compass.compass_sampling(DISTR, censor_time, tUk, tIn, Param, tIb, XPre[k], SPre[k])

                    if DISTR[0]:
                        Yn[k] = tYP
                    if DISTR[1]:
                        Yb[k] = tYB
                '''  Observation: Normal '''
                if observe_mode == 1:
                    CTk = (Ck * MCk[k]) @ xM
                    DTk = Dk
                    if obs_valid[k] == 2 and Param['censor_mode'] == 2:
                        # censor time
                        T = np.copy(Param['censor_time'])
                        if Param['censor_update_mode'] == 1:
                            # SPos Update first
                            Mx = CTk * XPre[k] + DTk @ In[k, :].T
                            Lx = np.maximum(EPS, norm.sf(Yn[k], loc=Mx, scale=np.sqrt(Vk)))
                            Gx = norm.pdf(Yn[k], Mx, np.sqrt(Vk))
                            Tx = Gx / Lx  # likelihood is not zero to avoid division by zero when computing Tx.
                            # Calculate SPos
                            Hx = (Yn[k] - Mx) / Vk
                            Sc = (CTk.T @ CTk) @ Tx * (Tx - Hx)
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                            # XPos update next
                            Ac = CTk.T @ Tx
                            XPos[k] = XPre[k] + SPos[k] * Ac
                        else:
                            in_loop = 10
                            # XPos update first
                            xpos = XPre[k]
                            for h in range(in_loop):
                                Mx = CTk @ xpos + DTk @ In[k, :].T
                                Lx = np.maximum(EPS, norm.sf(Yn[k], loc=Mx, scale=np.sqrt(Vk)))
                                Gx = norm.pdf(Yn[k], Mx, np.sqrt(Vk))
                                Tx = Gx / Lx
                                # update rule
                                Ac = CTk.T @ Tx
                                xpos = XPre[k] + SPre[k] * Ac

                            XPos[k] = xpos
                            Mx = CTk @ xpos + DTk @ In[k, :].T
                            Lx = np.maximum(EPS, norm.sf(Yn[k], loc=Mx, scale=np.sqrt(Vk)))
                            Gx = norm.pdf(Yn[k], Mx, np.sqrt(Vk))
                            Tx = Gx / Lx

                            # SPos update next
                            Hx = (Yn[k] - Mx) / Vk
                            Sc = (CTk.T @ CTk) @ Tx * (Tx - Hx)
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                    else:
                        # XPos
                        Sk = CTk @ SPre[k] @ CTk.T + Vk
                        Yp = CTk @ XPre[k] + DTk @ In[k, :].T
                        XPos[k] = XPre[k] + SPre[k] * CTk.T @ np.linalg.inv(Sk) @ (Yn[k] - Yp)
                        # SPos
                        SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + CTk.T @ (1 / Vk) @ CTk)
                ''' Observation: Bernoulli ,
                    For Bernoulli method, if there is any censored data it will be only based on resampling 
                    technique'''
                if observe_mode == 2:
                    ETk = (Ek * MEk[k]) @ xM
                    FTk = Fk
                    # XPos, SPos
                    # recursive mode
                    if update_mode == 1:
                        in_loop = 10
                        # XPos update
                        xpos = XPre[k]
                        for h in range(in_loop):
                            st = np.minimum(MAX_EXP, ETk @ xpos + FTk @ Ib[k, :])
                            pk = np.exp(st) / (1 + np.exp(st))
                            xpos = XPre[k] + SPre[k] * ETk.T @ (Yb[k] - pk)
                        XPos[k] = xpos
                        # SPos
                        SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + ETk.T @ np.diag(pk * (1 - pk)) @ ETk)

                    # one-step mode
                    if update_mode == 2:
                        st = np.minimum(MAX_EXP, ETk @ XPre[k] + FTk @ Ib[k, :].T)
                        pk = np.exp(st) / (1 + np.exp(st))
                        SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + ETk.T @ np.diag(pk * (1 - pk)) @ ETk)
                        XPos[k] = XPre[k] + SPos[k] * ETk.T @ (Yb[k] - pk)
                # Observation: Normal+Bernoulli
                if observe_mode == 3:
                    CTk = (Ck * MCk[k]) @ xM
                    DTk = Dk
                    ETk = (Ek * MEk[k]) @ xM
                    FTk = Fk
                    if obs_valid[k] == 2 and Param["censor_mode"] == 2:
                        # This is exactly the same for Normal distribution censor time
                        T = np.copy(Param["censor_time"])
                        # update mode 1
                        if Param["censor_update_mode"] == 1:
                            # SPos Update first
                            Mx = CTk @ XPre[k] + DTk @ In[k, :].T
                            Lx = np.maximum(EPS, norm.sf(Yn[k], loc=Mx, scale=np.sqrt(Vk)))
                            Gx = norm.pdf(Yn[k], Mx, np.sqrt(Vk))
                            Tx = Gx / Lx
                            # Update S
                            Hx = (Yn[k] - Mx) / Vk
                            Sc = (CTk.T @ CTk) @ Tx @ (Tx - Hx)
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                            # XPos Update next
                            Ac = CTk.T @ Tx
                            XPos[k] = XPre[k] + SPos[k] * Ac
                        else:
                            in_loop = 10
                            # XPos update first
                            xpos = XPre[k]
                            for h in range(in_loop):
                                Mx = CTk @ xpos + DTk @ In[k, :].T
                                Lx = np.maximum(EPS, norm.sf(Yn[k], loc=Mx, scale=np.sqrt(Vk)))
                                Gx = norm.pdf(Yn[k], Mx, np.sqrt(Vk))
                                Tx = Gx / Lx
                                # S update
                                Ac = CTk.T @ Tx
                                xpos = XPre[k] + SPre[k] * Ac
                            XPos[k] = xpos
                            Mx = CTk @ xpos + DTk @ In[k, :].T
                            Lx = np.maximum(EPS, norm.sf(T, loc=Mx, scale=np.sqrt(Vk)))
                            Gx = norm.pdf(T, Mx, np.sqrt(Vk))
                            Tx = Gx / Lx
                            # SPos update next
                            Hx = (Yn[k] - Mx) / Vk
                            Sc = (CTk.T @ CTk) @ Tx @ (Tx - Hx)
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                    else:
                        # XPos, SPos
                        # recursive mode
                        if update_mode == 1:
                            # recursive mode
                            in_loop = 10
                            xpos = XPre[k]
                            Yp = CTk @ XPre[k] + DTk @ In[k, :].T
                            Sk = (CTk.T @ np.linalg.inv(Vk) @ CTk + np.linalg.inv(SPre[k]))
                            for z in range(in_loop):
                                st = np.minimum(MAX_EXP, ETk @ xpos + FTk @ Ib[k, :].T)
                                pk = np.exp(st) / (1 + np.exp(st))
                                xpos = XPre[k] + np.linalg.inv(Sk) @ (
                                        ETk.T @ (Yb[k] - pk) + CTk.T @ np.linalg.inv(Vk) @ (Yn[k] - Yp))
                            XPos[k] = xpos
                            # SPos
                            SPos[k] = np.linalg.inv(
                                np.linalg.inv(SPre[k]) + CTk.T @ np.linalg.inv(Vk) @ CTk + ETk.T @ np.diag(
                                    pk * (1 - pk)) @ ETk)

                        # one-step mode
                        if update_mode == 2:
                            Yp = CTk @ XPre[k] + DTk @ In[k, :].T
                            st = np.minimum(MAX_EXP, ETk @ XPre[k] + FTk @ Ib[k, :].T)
                            pk = np.exp(st) / (1 + np.exp(st))
                            SPos[k] = np.linalg.inv(
                                np.linalg.inv(SPre[k]) + ETk.T @ np.diag(pk * (1 - pk)) @ ETk + CTk.T @ np.linalg.inv(
                                    Vk) @ CTk)
                            XPos[k] = XPre[k] + SPos[k] * (
                                    ETk.T @ (Yb[k] - pk) + CTk.T @ (Yn[k] - Yp) @ np.linalg.inv(Vk)
                            )
                # Observation: Gamma
                if observe_mode == 4:
                    CTk = (Ck * MCk[k]) @ xM
                    DTk = Dk

                    # this is exactly equal to Normal case
                    if obs_valid[k] == 2 and Param["censor_mode"] == 2:
                        # censor time
                        if Param["censor_update_mode"] == 1:
                            # expected y
                            Mx = np.exp(CTk @ XPre[k] + DTk @ In[k, :].T)
                            Hx = (Yn[k] - S) @ Vk / Mx
                            # components to estimate posterior
                            Lx = np.maximum(EPS, gammaincc(Vk, Hx))
                            Gx = gamma.pdf(Hx, Vk, loc=0, scale=1)
                            # temporary
                            Ta = Gx / Lx
                            # variace update
                            Sc = (CTk.T @ CTk) @ ((Vk - Hx) + Hx @ Ta) @ Hx @ Ta
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                            # XPos Update next
                            Ac = CTk.T @ Ta @ Hx
                            XPos[k] = XPre[k] + SPos[k] * Ac
                        else:
                            in_loop = 10
                            # XPos update first
                            xpos = XPre[k]
                            for h in range(in_loop):
                                # expected y
                                Mx = np.exp(CTk @ xpos + DTk @ In[k, :].T)
                                Hx = (Yn[k] - S) @ Vk / Mx
                                # components to estimate posterior
                                Lx = np.maximum(EPS, gammaincc(Vk, Hx))
                                Gx = gamma.pdf(Hx, Vk, loc=0, scale=1)
                                # temporary
                                Ta = Gx / Lx
                                # XPos Update next
                                Ac = CTk.T @ Ta @ Hx
                                xpos = XPre[k] + SPre[k] * Ac
                            XPos[k] = xpos
                            Mx = np.exp(CTk @ xpos + DTk @ In[k, :].T)
                            Hx = (Yn[k] - S) * Vk / Mx
                            # components to estimate posterior
                            Lx = np.maximum(EPS, gammaincc(Vk, Hx))
                            Gx = gamma.pdf(Hx, Vk, loc=0, scale=1)
                            # temporary
                            Ta = Gx / Lx
                            # variace update
                            Sc = (CTk.T @ CTk) @ ((Vk - Hx) + Hx @ Ta) @ Hx @ Ta
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                    else:
                        # XPos, SPos
                        # recursive mode
                        if update_mode == 1:
                            # recursive mode
                            Yk = Yn[k] - S
                            in_loop = 10
                            xpos = XPre[k]
                            for h in range(in_loop):
                                Yp = np.exp(CTk @ xpos + DTk @ In[k, :].T)
                                xpos = XPre[k] - SPre[k] * Vk @ CTk.T @ (1 - Yk / Yp)
                            XPos[k] = xpos
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + (Vk * (Yk / Yp)) * CTk.T @ CTk)
                        if update_mode == 2:
                            Yk = Yn[k] - S
                            Yp = np.exp(CTk @ XPre[k] + DTk @ In[k, :].T)
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + (Vk * (Yk / Yp)) * CTk.T @ CTk)
                            XPos[k] = XPre[k] - SPos[k] * Vk @ CTk.T @ (1 - Yk / Yp)
                # Observation: Gamma + Bernoulli
                if observe_mode == 6:
                    CTk = Ck * MCk[k] @ xM
                    DTk = Dk
                    ETk = Ek * MEk[k] @ xM
                    FTk = Fk
                    if obs_valid[k] == 2 and Param["censor_mode"] == 2:
                        # censor time
                        if Param["censor_update_mode"] == 1:
                            # expected y
                            Mx = np.exp(CTk @ XPre[k] + DTk @ In[k, :].T)
                            Hx = (Yn[k] - S) @ Vk / Mx
                            # components to estimate posterior
                            Lx = np.maximum(EPS, gammaincc(Vk, Hx))
                            Gx = gamma.pdf(Hx, Vk, loc=0, scale=1)
                            # temporary
                            Ta = Gx / Lx
                            # variance update
                            Sc = (CTk.T @ CTk) @ ((Vk - Hx) + Hx @ Ta) @ Hx @ Ta
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                            # XPos Update next
                            Ac = CTk.T @ Ta @ Hx
                            XPos[k] = XPre[k] + SPos[k] * Ac
                        else:
                            in_loop = 10
                            # XPos update first
                            xpos = XPre[k]
                            for h in range(in_loop):
                                # expected y
                                Mx = np.exp(CTk @ xpos + DTk @ In[k, :].T)
                                Hx = (Yn[k] - S) @ Vk / Mx
                                # components to estimate posterior
                                Lx = np.maximum(EPS, gammaincc(Vk, Hx))
                                Gx = gamma.pdf(Hx, Vk, loc=0, scale=1)
                                # temporary
                                Ta = Gx / Lx
                                # XPos Update next
                                Ac = CTk.T @ Ta @ Hx
                                xpos = XPre[k] + SPre[k] * Ac

                            XPos[k] = xpos
                            Mx = np.exp(CTk @ xpos + DTk @ In[k, :].T)
                            Hx = (Yn[k] - S) * Vk / Mx
                            # components to estimate posterior
                            Lx = np.maximum(EPS, gammaincc(Vk, Hx))
                            Gx = gamma.pdf(Hx, Vk, loc=0, scale=1)
                            # temporary
                            Ta = Gx / Lx
                            # variance update
                            Sc = (CTk.T @ CTk) @ ((Vk - Hx) + Hx @ Ta) @ Hx @ Ta
                            SPos[k] = np.linalg.inv(np.linalg.inv(SPre[k]) + Sc)
                    else:
                        # recursive mode
                        if update_mode == 1:
                            # XPos, SPos
                            in_loop = 10
                            Yk = Yn[k] - S
                            xpos = XPre[k]
                            for h in range(in_loop):
                                st = np.minimum(MAX_EXP, ETk @ xpos + FTk @ Ib[k, :].T)
                                pk = np.exp(st) / (1 + np.exp(st))
                                Yp = np.exp(CTk @ xpos + DTk @ In[k, :].T)
                                xpos = XPre[k] + SPre[k] * (ETk.T @ (Yb[k] - pk) - Vk @ CTk.T @ (1 - Yk / Yp))

                            XPos[k] = xpos
                            # SPos
                            SPos[k] = np.linalg.inv(
                                np.linalg.inv(SPre[k]) + CTk.T @ CTk @ Vk @ (Yk / Yp) + ETk.T @ ETk @ np.diag(
                                    pk * (1 - pk)))
                        # one-step mode
                        if update_mode == 2:
                            # XPos, SPos
                            Yk = Yn[k] - S
                            Yp = np.exp(CTk @ XPre[k] + DTk @ In[k, :].T)
                            # Pk
                            st = np.minimum(MAX_EXP, ETk @ XPre[k] + FTk @ Ib[k, :].T)
                            pk = np.exp(st) / (1 + np.exp(st))
                            # SPos
                            SPos[k] = np.linalg.inv(
                                np.linalg.inv(SPre[k]) + CTk.T @ CTk @ Vk @ (Yk / Yp) + ETk.T @ ETk @ np.diag(
                                    pk * (1 - pk)))
                            # XPos
                            XPos[k] = XPre[k] + SPos[k] * (ETk.T @ (Yb[k] - pk) - Vk @ CTk.T @ (1 - Yk / Yp))
            else:
                # randomly censored, the filter estimate will be equal to one-step prediction
                XPos[k] = XPre[k]
                SPos[k] = SPre[k]

        ''' Smoother Part - it is based on classical Kalman Smoother'''
        # Kalman Smoothing
        As = [np.array([[0]]) for _ in range(K)]  # [0] * K
        # posterior mean
        XSmt = [np.array([[0]]) for _ in range(K)]  # [0] * K
        XSmt[-1] = XPos[-1]
        # posterior variance
        SSmt = [np.array([[0]]) for _ in range(K)]  # [0] * K
        SSmt[-1] = SPos[-1]
        for k in range(K - 2, -1, -1):
            # Ak, equation (A.10)
            As[k] = SPos[k] @ Ak.T @ np.linalg.inv(SPre[k + 1])
            # Smting function, equation (A.9)
            XSmt[k] = XPos[k] + As[k] @ (XSmt[k + 1] - XPre[k + 1])
            # Variance update, equation (A.11)
            SSmt[k] = SPos[k] + As[k] @ (SSmt[k + 1] - SPre[k + 1]) @ As[k].T

        # Kalman smoother for time 0
        As0 = W0 @ Ak.T @ np.linalg.inv(SPre[0])
        XSmt0 = X0 + As0 @ (XSmt[0] - XPre[0])
        SSmt0 = W0 + As0 @ (SSmt[0] - SPre[0]) @ As0.T

        # Extra Component of the State Prediction Ckk = E(Xk*Xk)
        # Ckk = E(Xk*Xk) prediction by smoothing
        Ckk = [np.array([[0]]) for _ in range(K)]  # [0] * K
        for k in range(K):
            # Wk update - Smoothing Xk*Xk
            Ckk[k] = SSmt[k] + XSmt[k] @ XSmt[k].T
        Ckk0 = SSmt0 + XSmt0 @ XSmt0.T

        # Extra Component of the State Prediction Ckk = E(Xk*Xk-1)
        # Ckk_1=E(Xk-1*Xk) prediction by smoothing - it is kept at index K
        # Wkk_1= Ckk_1 + Bias
        # Covariance for smoothed estimates in state space models - Biometrica 1988-601-602
        Ckk_1 = [np.array([[0]]) for _ in range(K)]  # [0] * K
        Wkk_1 = [np.array([[0]]) for _ in range(K)]  # [0] * K
        for k in range(K):
            # Wkk update - Smoothing Xk-1*Xk
            if k > 0:
                Wkk_1[k] = As[k - 1] @ SSmt[k]
                Ckk_1[k] = Wkk_1[k] + XSmt[k] @ XSmt[k - 1].T
            else:
                Wkk_1[k] = As0 @ SSmt[k]
                Ckk_1[k] = Wkk_1[k] + XSmt[k] * XSmt0.T

        # Function return value
        rXSmt = XSmt
        rSSmt = SSmt
        rXPos = XPos
        rSPos = SPos
        rYb = Yb
        rYn = Yn

        """ Here, We Generate EYn and Yb Prediction"""
        # Generate EYn
        if DISTR[0] > 0:
            for k in range(K):
                # Filtering
                CTk = (Ck * MCk[k]) @ xM
                DTk = Dk
                # EYn
                if DISTR[0] == 1:
                    temp = CTk @ XSmt[k] + DTk @ In[k, :].T
                    EYn[k] = temp.T
                else:
                    temp = CTk @ XSmt[k] + DTk @ In[k, :].T
                    EYn[k] = np.exp(temp)

        if DISTR[1] == 1:
            for k in range(K):
                # Filtering
                ETk = (Ek * MEk[k]) @ xM
                FTk = Fk
                # YP
                temp = ETk @ XSmt[k] + FTk @ Ib[k, :].T
                EYb[k] = np.exp(temp.T) / (1 + np.exp(temp.T))

        """Parameter Estimation Section
        # Description of model parameters
        # Note that at index K of above variables, we keep:
        # 1. XPos(k) and SPos(k) are the optimum estimates of the Xk given 1:k
        # 2. XPre(k) and SPre(k) are the optimum estimates of the Xk give 1:k-1
        # 3. XSmt(k) and SSmt(k) are the optimum estimates of the Xk give 1:K
        # 4. Wk(k) is the E(Xk*Xk' given 1:K) observation
        # 5. Wkk(k) is the E(Xk-1*Xk' given 1:K) observation
        # note that Wkk(1), E(X0*X1)=X0*E(X1) as X0 is a deterministic term

        # Calculate State Transition Model Parameters
        # Update Ak and Bk - we assume iid noise terms
        """
        upAk = Ak
        upBk = Bk
        upWk = Wk
        # dimension of Xk
        dx = Ak.shape[1]
        # dimension of input
        di = Bk.shape[1]
        if Param["UpdateStateParam"] == 1:  # A and B is getting updated
            if Param["DiagonalA"] == 0:
                # Update the Ak, Bk row by row
                # We define At once
                At = np.zeros((dx + di, dx + di))
                for k in range(K):
                    if k == 0:
                        # Calculate At
                        SecA = Ckk0
                        SecB = XSmt0 @ Uk[k, :]
                        SecC = Uk[k, :].T @ XSmt0.T
                        SecD = Uk[k, :].T @ Uk[k, :]
                        At = At + np.vstack([np.hstack([SecA, SecB]), np.hstack([SecC, SecD])])
                    else:
                        # Calculate At
                        SecA = Ckk[k - 1]
                        SecB = XSmt[k - 1] * Uk[k, :]
                        SecC = Uk[k, :].T @ XSmt[k - 1].T
                        SecD = Uk[k, :].T @ Uk[k, :]
                        At = At + np.vstack([np.hstack([SecA, SecB]), np.hstack([SecC, SecD])])
                # We define Bt per row
                for d in range(dx):
                    # Build At, Bt
                    Bt = np.zeros((dx + di, 1))
                    for k in range(K):
                        # Calculate Bt
                        temp = Ckk_1[k]
                        SecA = temp[:, d]
                        temp = XSmt[k]
                        SecB = temp[d] @ Uk[k, :].T
                        Bt = Bt + np.vstack([SecA, SecB])
                    # Calculate At and Bt for the d-th row
                    T = np.linalg.pinv(At) @ Bt
                    upAk[d, :] = T[:dx]
                    upBk[d, :] = T[dx:].T
            if Param["DiagonalA"] == 1:
                # Update the Ak, Bk row by row
                for d in range(dx):
                    # Build At, Bt
                    At = np.zeros((1 + di, 1 + di))
                    Bt = np.zeros((1 + di, 1))
                    for k in range(K):
                        if k == 0:
                            # Calculate At
                            temp = Ckk0
                            SecA = temp[d, d]
                            temp = XSmt0
                            SecB = temp[d] @ Uk[k, :]
                            SecC = Uk[k, :].T @ temp[d]
                            SecD = Uk[k, :].T @ Uk[k, :]
                            At = At + np.vstack([np.hstack([SecA, SecB]), np.hstack([SecC, SecD])])
                        else:
                            # Calculate At
                            temp = Ckk[k - 1]
                            SecA = temp[d, d]
                            temp = XSmt[k - 1]
                            SecB = temp[d] @ Uk[k, :]
                            SecC = Uk[k, :].T @ temp[d]
                            SecD = Uk[k, :].T @ Uk[k, :]
                            At = At + np.vstack([np.hstack([SecA, SecB]), np.hstack([SecC, SecD])])
                        # Calculate Bt
                        temp = Ckk_1[k]
                        SecA = temp[d, d].T
                        temp = XSmt[k]
                        SecB = temp[d] * Uk[k, :].T
                        Bt = Bt + np.vstack([SecA, SecB])
                    # Calculate At and Bt for the d-th row
                    T = np.linalg.pinv(At) @ Bt
                    upAk[d, d] = T[0]
                    upBk[d, :] = T[1:]

                    #
        if Param["UpdateStateParam"] == 2:
            # A fixed, and B is getting updated
            # Update the Bk row by row
            # We define At once
            At = np.zeros((di, di))
            for k in range(K):
                # Calculate At
                SecD = np.outer(Uk[k, :], Uk[k, :])
                At += SecD

            # We define Bt per row
            for d in range(dx):
                # Build At, Bt
                Bt = np.zeros((di, 1))
                for k in range(K):
                    # Calculate Bt
                    if k == 0:
                        temp = XSmt[k] - Ak @ XSmt0
                    else:
                        temp = XSmt[k] - Ak @ XSmt[k - 1]
                    Bt += temp[d] @ Uk[k, :].T

                # Calculate At and Bt for the d-th row
                upBk[d, :] = np.linalg.pinv(At) @ Bt
        if Param["UpdateStateParam"] == 3:
            # A and B with specific structure
            # here, we find A and B for specific class of problem
            for d in range(0, upAk.shape[0], 2):  # odd components
                num_a = 0
                den_a = 0
                for k in range(0, K):
                    if k == 0:
                        num_a += SSmt0[d, d] + XSmt0[d] ** 2
                        num_a -= XSmt0[d] * 0
                        num_a -= Ckk0[d, d]
                        num_a += XSmt[k][d] * 0

                        den_a += SSmt0[d, d] + XSmt0[d] ** 2
                    else:
                        num_a += SSmt[k - 1][d, d] + (XSmt[k - 1][d]) ** 2
                        num_a -= XSmt[k - 1][d] @ Uk[k - 1, (d + 1) // 2]
                        num_a -= Ckk[k - 1][d, d]
                        num_a += XSmt[k][d] @ Uk[k - 1, (d + 1) // 2]

                        den_a += SSmt[k - 1][d, d] + (XSmt[k - 1][d] - Uk[k - 1, (d + 1) // 2]) ** 2
                upAk[d, d] = 1 - (num_a / den_a)
                upBk[d, (d + 1) // 2] = num_a / den_a

            for d in range(1, upAk.shape[0], 2):  # even components
                num_a = 0
                den_a = 0
                for k in range(0, K):
                    if k == 0:
                        num_a += SSmt0[d, d - 1] + XSmt0[d] * XSmt0[d - 1]
                        num_a -= XSmt0[d] * 0
                        num_a -= Ckk0[d, d - 1]
                        num_a += XSmt[k][d] * 0

                        den_a += SSmt0[d - 1, d - 1] + XSmt0[d - 1] ** 2
                    else:
                        num_a += SSmt[k - 1][d, d - 1] + XSmt[k - 1][d] * XSmt[k - 1][d - 1]
                        num_a -= XSmt[k - 1][d] * Uk[k - 1, d // 2]
                        num_a -= Ckk[k - 1][d, d - 1]
                        num_a += XSmt[k][d] * Uk[k - 1, d // 2]

                        den_a += SSmt[k - 1][d - 1, d - 1] + (XSmt[k - 1][d - 1] - sum(Uk[k - 1, :])) ** 2
                upAk[d, d] = 1
                upAk[d, d - 1] = -(num_a / den_a)
                upBk[d, d // 2] = num_a / den_a

            d = upAk.shape[0]
            num_a = 0
            den_a = 0
            for k in range(1, K + 1):
                if k == 1:
                    num_a = num_a + SSmt0[d, d] + XSmt0[d] ** 2
                    num_a = num_a - XSmt0[d] * 0
                    num_a = num_a - Ckk0[d, d]
                    num_a = num_a + XSmt[k][d] * 0

                    den_a = den_a + SSmt0[d, d] + XSmt0[d] ** 2
                else:
                    num_a = num_a + SSmt[k - 1][d, d] + XSmt[k - 1][d] ** 2
                    num_a = num_a - XSmt[k - 1][d] * sum(Uk[k - 1, :])
                    num_a = num_a - Ckk[k - 1][d, d]
                    num_a = num_a + XSmt[k][d] * sum(Uk[k - 1, :])

                    den_a = den_a + SSmt[k - 1][d, d] + (XSmt[k - 1][d] - sum(Uk[k - 1, :])) ** 2

            upAk[d, d] = 1 - (num_a / den_a)
            upBk[d, :] = num_a / den_a

        # Calculate Wk - we assume Wk is diagonal
        for d in range(dx):
            upWk[d, d] = 0
            for k in range(K):
                if k == 0:
                    # add E(Xk*Xk)
                    temp = Ckk[k]
                    upWk[d, d] = upWk[d, d] + temp[d, d]
                    # add E(Xk-1*Xk-1)*A^2
                    temp = Ckk0
                    upWk[d, d] = upWk[d, d] + upAk[d, :] @ temp @ upAk[d, :].T
                    # add Bk*Uk^2
                    upWk[d, d] = upWk[d, d] + (upBk[d, :] * Uk[k, :].T) ** 2
                    # add -2*A*E(Xk-1*Xk)
                    temp = Ckk_1[k]
                    upWk[d, d] = upWk[d, d] - upAk[d, :] @ (temp[:, d] + temp[d, :].T)
                    # add -2*B*U*E(Xk)
                    temp = XSmt[k]
                    upWk[d, d] = upWk[d, d] - 2 * (upBk[d, :] * Uk[k, :].T) @ temp[d]
                    # add 2 *B*U*A*E(Xk-1)
                    temp = XSmt0
                    upWk[d, d] = upWk[d, d] + 2 * (upBk[d, :] * Uk[k, :].T) * upAk[d, :] @ temp
                else:
                    # add E(Xk*Xk)
                    temp = Ckk[k]
                    upWk[d, d] = upWk[d, d] + temp[d, d]
                    # add E(Xk-1*Xk-1)*A^2
                    temp = Ckk[k - 1]
                    upWk[d, d] = upWk[d, d] + (upAk[d, :] @ (temp @ upAk[d, :].T))
                    # add Bk*Uk^2
                    upWk[d, d] = upWk[d, d] + (upBk[d, :] * Uk[k, :].T) ** 2
                    # add -2*A*E(Xk-1*Xk)
                    temp = Ckk_1[k]
                    upWk[d, d] = upWk[d, d] - upAk[d, :] @ (temp[:, d] + temp[d, :])
                    # add -2*B*U*E(Xk)
                    temp = XSmt[k]
                    upWk[d, d] = upWk[d, d] - 2 * (upBk[d, :] * Uk[k, :].T) @ temp[d]
                    # add 2 *B*U*A*E(Xk-1)
                    temp = XSmt[k - 1]
                    upWk[d, d] = upWk[d, d] + 2 * (upBk[d, :] * Uk[k, :].T) @ (upAk[d, :] @ temp)

            upWk[d, d] = upWk[d, d] / K
            # ---------------------------------
            # Update State parameters
            # ----------------------------------
            Ak = upAk
            Bk = upBk
            if Param["UpdateStateNoise"] == 1:
                Wk = upWk
        # Calculate the X0 parameters
        if Param["UpdateStateX0"] == 1:
            X0 = XSmt0
            W0 = Wk

        # Calculate likelihood function (Hidden Variable)
        # Constant terms are excluded
        MaxH = 0

        for d in range(Ak.shape[1]):
            # first variance
            MaxH = MaxH - 0.5 * K * np.log(Wk[d, d])
            # other terms
            TempH = 0
            for k in range(K):
                if k == 0:
                    # add E(Xk*Xk)
                    temp = Ckk[k]
                    TempH = TempH + temp[d, d]
                    # add E(Xk-1*Xk-1)*A^2
                    temp = Ckk0
                    TempH = TempH + Ak[d, :] @ temp @ Ak[d, :].T
                    # add Bk*Uk^2
                    TempH = TempH + (Bk[d, :] @ (Uk[k, :].T)) ** 2
                    # add -2*A*E(Xk-1*Xk)
                    temp = Ckk_1[k]
                    TempH = TempH - Ak[d, :] @ (temp[:, d] + temp[d, :])
                    # add -2*B*U*E(Xk)
                    temp = XSmt[k]
                    TempH = TempH - 2 * (Bk[d, :] @ Uk[k, :].T) * temp[d]
                    # add 2 *B*U*A*E(Xk-1)
                    temp = XSmt0
                    TempH = TempH + 2 * (Bk[d, :] @ Uk[k, :].T) * (Ak[d, :] @ temp)  # sumedh
                else:
                    # add E(Xk*Xk)
                    temp = Ckk[k]
                    TempH = TempH + temp[d, d]
                    # add E(Xk-1*Xk-1)*A^2
                    temp = Ckk[k - 1]
                    TempH = TempH + Ak[d, :] @ temp @ Ak[d, :].T
                    # add Bk*Uk^2
                    TempH = TempH + (Bk[d, :] @ Uk[k, :].T) ** 2
                    # add -2*A*E(Xk-1*Xk)
                    temp = Ckk_1[k]
                    TempH = TempH - Ak[d, :] @ (temp[:, d] + temp[d, :].T)
                    # add -2*B*U*E(Xk)
                    temp = XSmt[k]
                    TempH = TempH - 2 * ((Bk[d, :] @ Uk[k, :].T) * temp[d])  # sumedh
                    # add 2 *B*U*A*E(Xk-1)
                    temp = XSmt[k - 1]
                    TempH = TempH + 2 * (Bk[d, :] @ Uk[k, :].T) * (Ak[d, :] @ temp)  # sumedh

            MaxH = MaxH - 0.5 * TempH / Wk[d, d]

        # Update the Observation Model Parameters
        MaxO = 0
        if observe_mode == 1 or observe_mode == 3:
            # replace unobserved points with censored threshold
            # if we update all parameters of the model
            if Param["UpdateCModelParam"] == 1 and Param["UpdateCModelNoise"] == 1:
                if any(Param['cLinkUpdate']) or any(MDk[0]):
                    # generate index and matrixes for optimization
                    c_fill_ind = np.where(np.squeeze(Param['cLinkUpdate'][0]))[0]  # sumedh removed [0]
                    ck = Ck[0]
                    d_fill_ind = np.where(np.squeeze(MDk[0]))[0]  # sumedh removed [0]
                    dk = np.squeeze(Dk)
                    # initial parameters
                    # sumedh number of initial guess is larger than the lower bounds
                    p0 = np.concatenate((Vk[0], ck[c_fill_ind], dk[d_fill_ind]))

                    # define bounds
                    lower_bound = np.concatenate(([EPS], -1e3 * np.ones(len(p0) - 1)))
                    upper_bound = 1e3 * np.ones(len(p0))

                    # result_multi = minimize(
                    #     objFunc.normal_param_cdv.normal_param_cdv,  # The objective function to minimize (provided as
                    #     # a reference)
                    #     p0,  # The initial guess for optimization
                    #     args=(ck, dk, c_fill_ind, d_fill_ind, obs_valid, MCk, xM, Yn, XSmt, In, SSmt),
                    #     # Additional arguments
                    #     bounds=list(zip(lower_bound, upper_bound)),  # Bounds for optimization as a tuple
                    #     method='TNC'
                    # )
                    #
                    # p_opt = result_multi.x
                    # MaxO = -result_multi.fun

                    result_multi = compass.fminsearchbnd(objective_function.normal_param_cdv,
                                                     p0,
                                                     lower_bound,
                                                     upper_bound,
                                                     None,  # options
                                                     ck, dk, c_fill_ind, d_fill_ind, obs_valid, MCk, xM, Yn, XSmt, In,
                                                     SSmt)
                    p_opt = result_multi[0]
                    temp = result_multi[1]
                    MaxO = -temp

                    print([p_opt, MaxO])
                    print('Normal')

                    # put the estimates back to model
                    Vk[0] = p_opt[0]
                    Ck[0][c_fill_ind] = p_opt[1:1 + len(c_fill_ind)]
                    print(p_opt, MaxO)
                    Dk[0][d_fill_ind] = p_opt[1 + len(c_fill_ind):]
                else:
                    p0 = Vk[0]
                    # define bounds
                    # call optimization function
                    lower_bound = np.finfo(np.float64).eps * np.ones(1)
                    upper_bound = 1e3 * np.ones(1)
                    # call optimization function
                    result_multi = compass.fminsearchbnd(objective_function.normal_param_v,
                                                     p0,
                                                     lower_bound,
                                                     upper_bound,
                                                     None,  # options
                                                     Ck, Dk, obs_valid, MCk, xM, Yn, XSmt, In, SSmt)
                    p_opt = result_multi[0]
                    temp = result_multi[1]
                    MaxO = -temp
                    print('Normal')

                    ck = Ck
                    dk = Dk

                    # put the estimates back to model
                    Vk[0] = p_opt[0]

            # if we only update Parameters of the model
            if Param["UpdateCModelParam"] == 1 and Param["UpdateCModelNoise"] == 0:
                if len(np.nonzero(Param["cLinkUpdate"])[0]) != 0 or len(np.nonzero(MDk)[0]) != 0:
                    # initial parameters
                    c_fill_ind = np.where(np.squeeze(Param['cLinkUpdate'][0]))[0]  # sumedh removed [0]
                    ck = Ck[0]
                    d_fill_ind = np.where(np.squeeze(MDk[0]))[0]  # sumedh removed [0]
                    dk = np.squeeze(Dk)
                    p0 = np.concatenate((ck[c_fill_ind], dk[d_fill_ind]))
                    # define bounds
                    lower_bound = -1e3 * np.ones(len(p0))
                    upper_bound = 1e3 * np.ones(len(p0))
                    # call optimization function
                    result_multi = compass.fminsearchbnd(objective_function.normal_param_cd,
                                                     p0,
                                                     lower_bound,
                                                     upper_bound,
                                                     None,  # options
                                                     Vk, c_fill_ind, d_fill_ind, ck, dk, obs_valid, MCk, xM, Yn, XSmt,
                                                     In, SSmt)
                    p_opt = result_multi[0]
                    temp = result_multi[1]
                    MaxO = -temp
                    print('Normal')

                    sv = Vk[0]

                    # put the estimates back to the model
                    Ck[0][c_fill_ind] = p_opt[0:0 + len(c_fill_ind)]
                    Dk[0][d_fill_ind] = p_opt[0 + len(c_fill_ind):]
            # if we only update Noise Term
            if Param["UpdateCModelParam"] == 0 and Param["UpdateCModelNoise"] == 1:
                p0 = Vk[0]
                # define bounds

                # call optimization function
                lower_bound = np.finfo(np.float64).eps * np.ones(1)
                upper_bound = 1e3 * np.ones(1)
                # call optimization function
                result_multi = compass.fminsearchbnd(objective_function.normal_param_v,
                                                 p0,
                                                 lower_bound,
                                                 upper_bound,
                                                 None,  # options
                                                 Ck, Dk, obs_valid, MCk, xM, Yn, XSmt, In, SSmt)
                p_opt = result_multi[0]
                temp = result_multi[1]
                MaxO = -temp
                print('Normal')

                ck = Ck
                dk = Dk

                # put the estimates back to the model
                Vk[0] = p_opt[0]

        if observe_mode == 4 or observe_mode == 5:
            # continuous parameters update
            if Param["UpdateCModelParam"] == 1:
                if Param["UpdateCModelNoise"] and Param["UpdateCModelShift"]:  # Full model update
                    # keep learning param
                    c_fill_ind = np.where(np.squeeze(Param['cLinkUpdate'][0]))[0]  # sumedh removed [0]
                    ck = Ck[0]
                    d_fill_ind = np.where(np.squeeze(MDk[0]))[0]  # sumedh removed [0]
                    dk = np.squeeze(Dk)
                    # initiate parameters
                    p0 = np.concatenate((Vk[0],[S], ck[c_fill_ind], dk[d_fill_ind]))
                    #p0 = [Vk, S, list(ck[c_fill_ind]), list(dk[d_fill_ind])]
                    # lower and upper bound (dispersion Shift Ck Dk)
                    lower_bound = np.concatenate(([1], [0], -1e3 * np.ones_like(c_fill_ind), -1e3 * np.ones_like(d_fill_ind)))
                    upper_bound = np.concatenate(([50], [0.99 * np.min(Yn)], 1e3 * np.ones_like(c_fill_ind), 1e3 * np.ones_like(d_fill_ind)))
                    # call optimization function
                    # options = optimoptions('lsqnonlin', 'Display', 'off', 'DiffMaxChange', 100, 'MaxIter', 1000)
                    # call optimization function
                    result_multi = compass.fminsearchbnd(objective_function.gamma_param_full,
                                                     p0,
                                                     lower_bound,
                                                     upper_bound,
                                                     None,  # options
                                                     Yn, obs_valid, ck, dk, MCk, xM, XSmt, SSmt, In, c_fill_ind,
                                                     d_fill_ind)
                    p_opt = result_multi[0]
                    temp = result_multi[1]
                    MaxO = -temp
                    print('Gamma')
                    # update param
                    Vk[0] = p_opt[0]
                    S = p_opt[1]
                    Ck[0][c_fill_ind] = p_opt[1:2 + len(c_fill_ind)]
                    print(p_opt, MaxO)
                    Dk[0][d_fill_ind] = p_opt[2 + len(c_fill_ind):]

                elif Param["UpdateCModelNoise"]:  # Update Ck,Dk, plus V
                    # keep learning param
                    c_fill_ind = np.where(np.squeeze(Param['cLinkUpdate'][0]))[0]  # sumedh removed [0]
                    ck = Ck[0]
                    d_fill_ind = np.where(np.squeeze(MDk[0]))[0]  # sumedh removed [0]
                    dk = np.squeeze(Dk)
                    # initial parameters
                    # sumedh number of initial guess is larger than the lower bounds
                    p0 = np.concatenate((Vk[0], ck[c_fill_ind], dk[d_fill_ind]))

                    # lower and upper bound (dispersion Shift Ck Dk)
                    lower_bound = np.concatenate(
                        ([1], -1e3 * np.ones_like(c_fill_ind), -1e3 * np.ones_like(d_fill_ind)))
                    upper_bound = np.concatenate(([50], 1e3 * np.ones_like(c_fill_ind), 1e3 * np.ones_like(d_fill_ind)))
                    # call optimization function
                    result_multi = compass.fminsearchbnd(objective_function.gamma_param_minus_S,
                                                     p0,
                                                     lower_bound,
                                                     upper_bound,
                                                     None,  # options
                                                     ck, dk, c_fill_ind, d_fill_ind, obs_valid, MCk, xM, Yn, XSmt, In,
                                                     SSmt, S)
                    p_opt = result_multi[0]
                    temp = result_multi[1]
                    MaxO = -temp
                    print('Gamma')
                    # update param
                    Vk[0] = p_opt[0]
                    Ck[0][c_fill_ind] = p_opt[1:1 + len(c_fill_ind)]
                    print(p_opt, MaxO)
                    Dk[0][d_fill_ind] = p_opt[1 + len(c_fill_ind):]

                elif Param["UpdateCModelShift"]:
                    # Update Ck, Dk, plus Shift
                    # keep learning param
                    c_fill_ind = np.where(np.squeeze(Param['cLinkUpdate'][0]))[0]  # sumedh removed [0]
                    ck = Ck[0]
                    d_fill_ind = np.where(np.squeeze(MDk[0]))[0]  # sumedh removed [0]
                    dk = np.squeeze(Dk)
                    # initial parameters
                    # sumedh number of initial guess is larger than the lower bounds
                    p0 = np.concatenate(([S], ck[c_fill_ind], dk[d_fill_ind]))
                    # lower and upper bound (dispersion Shift Ck Dk)
                    lower_bound = np.concatenate(
                        ([0], -1e3 * np.ones_like(c_fill_ind), -1e3 * np.ones_like(d_fill_ind)))
                    upper_bound = np.concatenate(
                        ([0.99 * np.min(Yn)], 1e3 * np.ones_like(c_fill_ind), 1e3 * np.ones_like(d_fill_ind)))
                    result_multi = compass.fminsearchbnd(objective_function.gamma_param_minus_v,
                                                     p0,
                                                     lower_bound,
                                                     upper_bound,
                                                     None,  # options
                                                     Yn, Vk, ck, dk, MCk, xM, In, XSmt, SSmt, c_fill_ind, d_fill_ind,
                                                     obs_valid)
                    p_opt = result_multi[0]
                    temp = result_multi[1]
                    MaxO = -temp
                    print('Gamma')
                    # update param
                    S = p_opt[0]
                    Ck[0][c_fill_ind] = p_opt[1:1 + len(c_fill_ind)]
                    print(p_opt, MaxO)
                    Dk[0][d_fill_ind] = p_opt[1 + len(c_fill_ind):]
                else:
                    # keep learning param
                    c_fill_ind = np.where(np.squeeze(Param['cLinkUpdate'][0]))[0]  # sumedh removed [0]
                    ck = Ck[0]
                    d_fill_ind = np.where(np.squeeze(MDk[0]))[0]  # sumedh removed [0]
                    dk = np.squeeze(Dk)
                    # initiate p0
                    p0 = np.concatenate((ck[c_fill_ind], dk[d_fill_ind]))
                    if np.any(p0):
                        # lower and upper bound (dispersion Shift Ck Dk)
                        lower_bound = np.concatenate((-1e3 * np.ones(len(c_fill_ind)), -1e3 * np.ones(len(d_fill_ind))))
                        upper_bound = np.concatenate((1e3 * np.ones(len(c_fill_ind)), 1e3 * np.ones(len(d_fill_ind))))
                        # call optimization function
                        result_multi = compass.fminsearchbnd(objective_function.gamma_param_cd,
                                                         p0,
                                                         lower_bound,
                                                         upper_bound,
                                                         None,  # options
                                                         Yn, S, Vk, ck, c_fill_ind, dk, d_fill_ind, obs_valid, MCk, xM,
                                                         XSmt, SSmt, In)
                        p_opt = result_multi[0]
                        temp = result_multi[1]
                        MaxO = -temp
                        print('Gamma')
                        # update param
                        Ck[0][c_fill_ind] = p_opt[:len(c_fill_ind)]
                        print(p_opt, MaxO)
                        Dk[0][d_fill_ind] = p_opt[0 + len(c_fill_ind):]

        #  Estimate Discrete Components
        MaxB = 0
        if DISTR[1] == 1:
            if Param["UpdateDModelParam"] == 1:
                if np.any(Param["dLinkUpdate"]) or np.any(MFk):
                    # generate index and matrices for optimization
                    e_fill_ind = np.where(np.squeeze(Param["dLinkUpdate"]))[0]
                    ek = Ek[0]
                    f_fill_ind = np.where(np.squeeze(MFk))
                    fk = np.squeeze(Fk)
                    # initial parameters
                    p0 = np.concatenate([ek[e_fill_ind], fk[f_fill_ind]])
                    # define bounds
                    lower_bound = -1e3 * np.ones(len(p0))
                    upper_bound = 1e3 * np.ones(len(p0))

                    # call optimization function
                    # f = bernoulli_param(p0, e_fill_ind, f_fill_ind, ek, fk, obs_valid, MEk, xM, XSmt, SSmt, Ib, Yb)
                    result_multi = compass.fminsearchbnd(objective_function.bernoulli_param,
                                                     p0,
                                                     lower_bound,
                                                     upper_bound,
                                                     None,  # options
                                                     e_fill_ind, f_fill_ind, ek, fk, obs_valid, MEk, xM, XSmt, SSmt,
                                                     Ib, Yb)

                    p_opt = result_multi[0]
                    temp = result_multi[1]
                    MaxB = -temp
                    print('Bernoulli')
                    print([p_opt, MaxB])
                    # put the estimates back to model
                    Ek[e_fill_ind] = p_opt[:len(e_fill_ind)] if len(p_opt[:len(e_fill_ind)]) > 0 else None
                    Fk[0][f_fill_ind] = p_opt[len(e_fill_ind):] if len(p_opt[len(e_fill_ind):]) > 0 else None
        ML[iter]['Total'] = MaxH + MaxO + MaxB
        ML[iter]['State'] = MaxH
        ML[iter]['ObsrvNormGamma'] = MaxO
        ML[iter]['ObsrvBern'] = MaxB
    # Update Model Parameters
    # State Transition Model Parameters
    Param['Ak'] = Ak
    Param['Bk'] = Bk
    Param['Wk'] = Wk
    Param['X0'] = X0
    Param['W0'] = W0

    # Binary section parameters
    if DISTR[1] > 0:
        Param['Ek'] = Ek
        Param['Fk'] = Fk

    # Continuous section parameters
    if DISTR[0] > 0:
        Param['Ck'] = Ck
        Param['Dk'] = Dk
        Param['Vk'] = Vk
        if DISTR[0] == 2:
            Param['S'] = S

    return XSmt, SSmt, Param, XPos, SPos, ML, EYn, EYb, rYn, rYb
