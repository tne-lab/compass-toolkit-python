"""
Translation of MATLAB code 'compass_Qk.m' to python
17 Nov 2022  Translated by Jazlin Taylor
Translation of MATLAB code 'compass_filtering.m' to python
7th Dec 2022 Translated, Tested & Modified by Sumedh Nagrale

15th Sep 2023
Following modifications were made to
1. Removing bugs
"""

import numpy as np
import compass_Tk as Ctk
import compass_Qk as Cqk
import compass_sampling as Csam
import algorithms_details as ad

"""
Comments for the variables down here
"""


def compass_filtering(DISTR=None, Uk=None, In=None, Param=None, Ib=None, Yn=None, Yb=None, obs_valid=None, XPos0=None,
                      SPos0=None):
    SP = []
    # variables from the dictionary to reduce the clutter down in the main code
    update_mode = np.copy(Param['UpdateMode'])
    # State Space Model(X(k + 1) = Ak * X(k) + Bk * Uk + Wk * iid white noise)
    Ak = np.copy(Param['Ak'])
    Bk = np.copy(Param['Bk'])
    Wk = np.copy(Param['Wk'])
    xM = np.copy(Param['xM'])
    # Uk = np.copy(Param['Uk'])

    # Observation mode: 1 to 5
    if DISTR[0] == 1:
        observe_mode = DISTR[0] + 2 * DISTR[1]
    elif DISTR[0] == 2:
        observe_mode = 2 * DISTR[0] + DISTR[1]
    else:
        observe_mode = 2 * DISTR[1]

    # Build Mask Ck, Dk ,EK and Fk - note that Ck, Ek are time-dependent and the Dk and Fk is linked to a subset of
    # Input
    [MCk, MDk] = Ctk.compass_Tk(In, Param)
    if DISTR[1] == 1:
        [MEk, MFk] = Cqk.compass_Qk(Ib, Param)

    # Censored Reaction Time
    # Remember this is with respect to the documentation where we have the obs_valid value as 2,
    # this makes it to be with respect to the idea of the gamma log distributions
    if 2 in obs_valid:
        censor_time = np.copy(Param['censor_time'])

    ''' Normal/Gamma Observation Model'''
    if DISTR[0] > 0:
        '''
        % For Normal,  Y(k)=(Ck.*Tk)*X(k)+Dk*Ik + Vk    Vk variance of iid white noise
        % For Gamma,   Y(k)=exp((Ck.*Tk)*X(k)+Dk*Ik)    Vk is dispersion term 
        % ------------------
        % Y(k) is the observation, and Ik is the input either indicator or continuous
        % Ck, Dk, Vk are the model parameters
        % Tk is model specific function - it is original set to but a one matrix
        % ------------------
        % Ck, 1xM matrix - (Y is an scalar observation at each time point ... - The Tk has the same size of input, 
        % and it is specifically designed for our task. It can be set to all 1 matrix)
        '''
        Ck = np.copy(Param['Ck'])
        # Bk, NxS3 matrix - (We have an input of the length S3, and Dk will be size of NxS3)
        Dk = Param['Dk'] * MDk
        # Vk, is scaler representing noise in Normal or Dispresion Term in Gamma
        Vk = np.copy(Param["Vk"])

    '''Binary Observation Model (P(k)=sigmoid((Ek.*Qk)*X(k)+Fk*Ik) )'''
    if DISTR[1] == 1:
        '''
        % ------------------
        % P(k) is the observation probability at time k, and Ik is the input either indicator or continuous
        % Ek, and Fk are the model parameters
        % Qk is model specific function - it is original set to but a one matrix
        % ------------------
        % Ck, NxM matrix - similar to Ck, Tk
        '''
        Ek = np.copy(Param['Ek'])
        # Fk, NxS5 matrix - Similar to Dk
        Fk = Param['Fk'] * MFk

    '''Check Uk'''
    if np.any(Uk):
        Uk = np.zeros((1, Bk.shape[1]))

    ''' 
    Filter 
    One step prediction
    '''
    XPre = Ak @ XPos0 + Bk @ Uk.T
    SPre = Ak @ SPos0 @ Ak.T + Wk
    '''Check if the data point is censored or not'''
    if obs_valid >= 1:
        '''Draw a sample if it is censored'''
        if obs_valid == 2:
            [tYP, tYB] = Csam.compass_sampling(DISTR=DISTR, Cut_Time=censor_time, tUk = Uk, tIn=In, tParam=Param, Ib=Ib,
                                               XPos0=XPre, SPos0=SPre)
            if DISTR[0] > 0:
                Yn = tYP
            elif DISTR[1] == 1:
                Yb = tYB

        # Observation: Normal
        if observe_mode == 1:
            CTk = (Ck * MCk[0]) @ xM
            DTk = Dk
            # XPos
            Sk = CTk @ SPre @ CTk.T + Vk
            Yp = CTk @ XPre + DTk @ In.T
            XPos = XPre + (SPre @ CTk.T @ np.linalg.inv(Sk) @ (Yn - Yp))
            # SPos
            SPos = np.linalg.inv(np.linalg.inv(SPre) + CTk.T @ np.linalg.inv(Vk) @ CTk)

        ''' Observation: Bernoulli '''
        if observe_mode == 2:
            ETk = (Ek * MEk) @ xM
            FTk = Fk
            ''' XPos, SPos: recursive mode '''
            if update_mode == 1:
                in_loop = 10
                # XPos update
                xpos = XPre
                for t in range(in_loop):
                    st = ETk @ xpos + FTk @ Ib.T
                    pk = np.exp(st) / (1 + np.exp(st))
                    xpos = XPre + SPre @ ETk.T @ (Yb - pk)
                XPos = xpos
                # SPos
                SPos = np.linalg.inv(np.linalg.inv(SPre) + ETk.T @ np.diag(pk * (1 - pk)) @ ETk)
            #  one-step mode
            if update_mode == 2:
                st = ETk @ XPre + FTk @ Ib.T
                pk = np.exp(st) / (1 + np.exp(st))
                SPos = np.linalg(np.linalg.inv(SPre) + ETk.T @ np.diag(pk * (1 - pk) @ ETk))
                XPos = XPre + SPos @ ETk.T @ (Yb - pk)

        ''' Observation: Normal+Bernoulli'''
        if observe_mode == 3:
            CTk = (Ck * MCk[0]) @ xM
            DTk = Dk
            ETk = (Ek * MEk[0]) @ xM
            FTk = Fk
            # XPos, SPos
            # Recursive mode
            if update_mode == 1:
                # recursive mode
                in_loop = 10
                xpos = XPre
                Yp = CTk @ XPre + DTk @ In.T
                Sk = CTk.T @ np.linalg.inv(Vk) @ CTk + np.linalg.inv(SPre)
                for t in range(in_loop):
                    st = ETk @ xpos + FTk @ Ib.T
                    pk = np.exp(st) / (1 + np.exp(st))
                    xpos = XPre + np.linalg.inv(Sk) @ (ETk.T * (Yb - pk) + CTk.T @ np.linalg.inv(Vk) @ (Yn - Yp))
                XPos = xpos
                # SPos
                SPos = np.linalg.inv(
                    np.linalg.inv(SPre) + CTk.T @ np.linalg.inv(Vk) @ CTk + ETk.T @ np.diag(pk * (1 - pk)) @ ETk)
            # one-step mode
            if update_mode == 2:
                Yp = CTk @ XPre + DTk @ In.T
                st = ETk @ XPre + FTk @ Ib.T
                pk = np.exp(st) / (1 + np.exp(st))
                SPos = np.linalg.inv(
                    (np.linalg.inv(SPre) + ETk.T @ np.diag(pk * (1 - pk)) @ ETk + CTk.T @ np.linalg.inv(Vk) @ CTk))
                XPos = XPre + SPos @ (ETk.T @ (Yb - pk) + CTk.T @ (Yn - Yp) @ np.linalg.inv(Vk))

        ''' Observation: Gamma '''
        if observe_mode == 4:
            CTk = (Ck * MCk[0]) @ xM
            DTk = Dk
            # XPos, SPos
            # recursive mode
            if update_mode == 1:
                # recursive mode
                Yk = Yn - Param['S']
                in_loop = 10
                xpos = XPre
                for t in range(in_loop):
                    Yp = np.exp(CTk @ xpos + DTk @ In.T)
                    xpos = XPre - SPre @ Vk @ CTk.T @ (1 - Yk * np.linalg.inv(Yp))
                XPos = xpos
                SPos = np.linalg.inv((np.linalg.inv(SPre) + (Vk @ (Yk  * np.linalg.inv(Yp))) * CTk.T @ CTk))
            if update_mode == 2:
                Yk = Yn - Param['S']
                Yp = np.exp(CTk @ XPre + DTk @ In.T)
                SPos = np.linalg.inv(np.linalg.inv(SPre) + Vk @ (Yk @ np.linalg.inv(Yp)) @ Ctk.T @ CTk)
                XPos = XPre - SPos @ Vk @ CTk.T @ (1 - (Yk @ np.linalg.inv(Yp)))

        ''' Observation: Gamma+Bernoulli'''
        if observe_mode == 5:
            CTk = (Ck @ MCk[0]) @ xM
            DTk = Dk
            ETk = (Ek * MEk[0]) @ xM
            FTk = Fk
            # recursive mode
            if update_mode == 1:
                # XPos, SPos
                in_loop = 10
                Yk = Yn - Param['S']
                xpos = XPre
                for t in range(in_loop):
                    st = ETk @ xpos + FTk @ Ib.T
                    pk = np.exp(st) / (1 + np.exp(st))
                    Yp = np.exp(CTk @ xpos + DTk @ In.T)
                    xpos = XPre + SPre @ (ETk.T @ (Yb - pk) - Vk @ CTk.T @ (1 - Yk * np.linalg.inv(Yp)))
                XPos = xpos
                SPos = np.linalg.inv(
                    np.linalg.inv(SPre) + CTk.T @ CTk @ Vk @ (Yk * np.linalg.inv(Yp)) + ETk.T @ ETk @ np.diag(
                        pk * (1 - pk)))
            if update_mode == 2:
                # XPos, SPos
                Yk = Yn - Param['S']
                Yp = np.exp(CTk @ XPre + DTk @ In.T)
                # Pk
                st = ETk @ XPre + FTk @ Ib.T
                pk = np.exp(st) / (1 + np.exp(st))
                # SPos
                SPos = np.linalg.inv(
                    np.linalg.inv(SPre) + CTk.T @ CTk @ Vk @ (Yk @ np.linalg.inv(Yp)) + ETk.T @ ETk @ np.diag(
                        pk * (1 - pk)))
                # XPos
                XPos = XPre + SPre @ (ETk.T @ (Yb - pk) - Vk @ CTk.T @ (1 - Yk @ np.linalg.inv(Yp)))
    else:
        # randomly censored, the filter estimate will be equal to one-step prediction
        XPos = XPre
        SPos = SPre

    # Update prediction
    YP = []
    YB = []
    # SP = np.empty(1, dtype=object)

    if DISTR[0] > 0:
        # Filtering
        CTk = (Ck * MCk[0]) @ xM
        DTk = Dk
        # EYn
        if DISTR[0] == 1:
            temp = CTk @ XPos + DTk @ In.T
            YP = temp.T
        else:
            temp = CTk @ XPos + DTk @ In.T
            YP = np.exp(0.5 * CTk @ temp @ CTk.T) # np.exp(temp) @ np.exp(0.5 @ CTk @ SPos @ CTk.T)
            # SP = np.exp(2 * temp) @ np.exp(2 @ CTk @ SPos @ CTk.T) - YP @ YP
            # YP = np.exp(0.5 @ CTk @ SPos @ CTk.T)
            YP = Param['S'] + YP

    if DISTR[1] == 1:
        # Filtering
        ETk = (Ek * MEk[0]) @ xM
        FTk = Fk
        # YP
        temp = ETk @ XPos + FTk @ Ib.T
        YB = np.exp(temp.T) / (1 + np.exp(temp.T))

    return XPos, SPos, YP, YB
