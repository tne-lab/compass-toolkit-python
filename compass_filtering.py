"""
Translation of MATLAB code 'compass_Qk.m' to python
17 Nov 2022  Translated by Jazlin Taylor
Translation of MATLAB code 'compass_filtering.m' to python
7th Dec 2022 Translated, Tested & Modified by Sumedh Nagrale
"""

import numpy as np
import compass_Tk as Ctk
import compass_Qk as Cqk
import compass_sampling as Csam


def compass_filtering(DISTR=None, tIn=None, tParam=None, Ib=None, Yn=None, obs_valid=None, XPos0=None, SPos0=None):
    update_mode = tParam['UpdateMode']
    if DISTR[0] >= 1:
        observe_mode = DISTR[0] + 2 * DISTR[1]
    elif DISTR[0] == 2:
        observe_mode = 2 * DISTR[0] + DISTR[1]
    else:
        observe_mode = 2 * DISTR[1]
    MCk, MDk = Ctk.compass_Tk(tIn, tParam)
    if DISTR[1] == 1:
        MEk, MFk = Cqk.compass_Qk(tIn, tParam)
    Ak = tParam['Ak']
    Bk = tParam['Bk']
    Wk = tParam['Wk']
    xM = tParam['xM']

    if (np.argwhere(obs_valid == 2)).size != 0:
        censor_time = tParam['censor_time']
    if DISTR[0] > 0:
        Ck = tParam['Ck']
        Dk = tParam['Dk'] * MDk
        Vk = tParam['Vk']
    if DISTR[1] == 1:
        Ek = tParam['Ek']
        Fk = tParam['Fk'] * MFk
    if (tParam['Uk']).size != 0:
        Uk = np.zeros((1, np.size(Bk, axis=1)))
    # Filter
    XPre = Ak @ XPos0 + Bk @ Uk.T
    SPre = Ak @ SPos0 @ Ak.T + Wk
    censor_time = tParam['censor_time']
    if obs_valid >= 1:
        if obs_valid == 2:
            tYP, tYB = Csam.compass_sampling(DISTR=DISTR, Cut_Time=censor_time, tIn=tIn, tParam=tParam, Ib=Ib,
                                             XPos0=XPre, SPos0=SPre)
            if DISTR[0] > 0:
                Yn = tYP
            elif DISTR[1] == 1:
                Yb = tYB
        if observe_mode == 1:
            CTk = (Ck * MCk[0]) @ xM
            DTk = Dk
            Sk = CTk @ SPre @ CTk.T + Vk
            Yp = CTk @ XPre + DTk @ tIn.T
            XPos = XPre + (SPre @ CTk.T @ np.linalg.inv(Sk) @ (Yn - Yp))
            SPos = np.linalg.inv((SPre) + CTk.T @ np.linalg.inv(Vk) @ CTk)
        if observe_mode == 2:
            ETk = (Ek * MEk) @ xM
            FTk = Fk
            if update_mode == 1:
                in_loop = 10
                xpos = XPre
                for t in range(in_loop):
                    st = ETk @ xpos + FTk @ Ib.T
                    pk = np.exp(st) / (1 + np.exp(st))
                    xpos = XPre + SPre @ ETk.T @ (Yb - pk)
            XPos = xpos
            SPos = np.linalg(np.linalg.inv(SPre) + ETk.T @ np.diag(pk @ (1 - pk)) @ ETk)
            if update_mode == 2:
                st = ETk @ XPre + FTk @ Ib.T
                pk = np.exp(st) / (1 + np.exp(st))
                SPos = np.linalg(np.linalg.inv(SPre) + ETk.T @ np.diag(pk * (1 - pk) @ ETk))
                XPos = XPre + SPos @ ETk.T @ (Yb - pk)
            if update_mode == 3:
                CTk = (Ck * MCk[0]) @ xM
                DTk = Dk
                ETk = (Ek * MEk[0]) @ xM
                FTk = Fk
                if update_mode == 1:
                    in_loop = 10
                    xpos = XPre
                    Yp = CTk @ XPre + DTk @ tIn.T
                    Sk = CTk.T @ np.linalg.inv(Vk) @ CTk + np.linalg.inv(SPre)
                    for t in range(in_loop):
                        st = ETk @ xpos + FTk @ Ib.T
                        pk = np.exp(st) / (1 + np.exp(st))
                        xpos = XPre + np.linalg.inv(Sk) @ (ETk.T * (Yb - pk) + CTk.T @ np.linalg.inv(Vk) @ (Yn - Yp))
                    XPos = xpos
                    SPos = np.linalg.inv(
                        np.linalg.inv(SPre) + CTk.T @ np.linalg.inv(Vk) @ CTk + ETk.T @ np.diag(pk * (1 - pk)) @ ETk)
                if update_mode == 2:
                    Yp = CTk @ XPre + DTk @ tIn.T
                    st = ETk @ XPre + FTk @ Ib.T
                    pk = np.exp(st) / (1 + np.exp(st))
                    SPos = np.linalg.inv(
                        (np.linalg.inv(SPre) + ETk.T @ np.diag(pk * (1 - pk)) @ ETk + CTk.T @ np.linalg.inv(Vk) @ CTk))
                    XPos = XPre + SPos @ (ETk.T @ (Yb - pk) + CTk.T @ (Yn - Yp) @ np.linalg.inv(Vk))
            if observe_mode == 4:
                CTk = (Ck * MCk[0]) * xM
                DTk = Dk
                if update_mode == 1:
                    Yk = Yn - tParam['S']
                    in_loop = 10
                    xpos = XPre
                    for t in range(in_loop):
                        Yp = np.exp(CTk @ xpos + DTk @ tIn.T)
                        xpos = XPre - SPre @ Vk @ CTk.T @ (1 - Yk @ np.linalg.inv(Yp))
                    XPos = xpos
                    SPos = np.linalg.inv((np.linalg.inv(SPre) + (Vk @ (Yk @ np.linalg.inv(Yp))) * CTk.T @ CTk))
                if update_mode == 2:
                    Yk = Yn - tParam['S']
                    Yp = np.exp(CTk @ XPre + DTk @ tIn.T)
                    SPos = np.linalg.inv(np.linalg.inv(SPre) + Vk @ (Yk @ np.linalg.inv(Yp)) @ Ctk.T @ CTk)
                    XPos = XPre - SPos @ Vk @ CTk.T @ (1 - Yk @ np.linalg.inv(Yp))
            if observe_mode == 5:
                CTk = (Ck @ MCk[0]) @ xM
                DTk = Dk
                ETk = (Ek @ MEk[0]) @ xM
                FTk = Fk
                if update_mode == 1:
                    in_loop = 10
                    Yk = Yn - tParam['S']
                    xpos = XPre
                    for t in range(in_loop):
                        st = ETk @ xpos + FTk @ Ib.T
                        pk = np.exp(st) / (1 + np.exp(st));
                        Yp = np.exp(CTk @ xpos + DTk @ tIn.T)
                        xpos = XPre + SPre @ (ETk.T @ (Yb - pk) - Vk @ CTk.T @ (1 - Yk @ np.linalg.inv(Yp)))
                    XPos = xpos
                    SPos = np.linalg.inv(
                        np.linalg.inv(SPre) + CTk.T @ CTk @ Vk @ (Yk @ np.linalg.inv(Yp)) + ETk.T @ ETk @ np.diag(pk * (1 - pk)))
                if update_mode == 2:
                    Yk = Yn - tParam['S']
                    Yp = np.exp(CTk @ XPre + DTk @ tIn.T)
                    st = ETk @ XPre + FTk @ Ib.T
                    pk = np.exp(st) / (1 + np.exp(st))
                    SPos = np.linalg.inv(
                        np.linalg.inv(SPre) + CTk.T @ CTk @ Vk @ (Yk @ np.linalg.inv(Yp)) + ETk.T @ ETk @ np.diag(pk * (1 - pk)))
                    XPos = XPre + SPre @ (ETk.T @ (Yb - pk) - Vk @ CTk.T @ (1 - Yk @ np.linalg.inv(Yp)))
    else:
        XPos = XPre
        SPos = SPre
    YP = np.empty(1, dtype=object)
    YB = np.empty(1, dtype=object)
    SP = np.empty(1, dtype=object)
    if DISTR[0] > 0:
        CTk = (Ck * MCk[0]) @ xM
        DTk = Dk
        if DISTR[0] == 1:
            temp = CTk @ XPos + DTk @ tIn.T
            YP = temp.T
        else:
            temp = CTk @ XPos + DTk @ tIn.T
            YP = np.exp(temp) @ np.exp(0.5 @ CTk @ SPos @ CTk.T)
            SP = np.exp(2 * temp) @ np.exp(2 @ CTk @ SPos @ CTk.T) - YP @ YP
            YP = tParam['S'] + YP

    if DISTR[1] == 1:
        ETk = (Ek * MEk[0]) @ xM
        FTk = Fk
        temp = ETk @ XPos + FTk @ Ib.T
        YB = np.exp(temp.T)/(1+np.exp(temp.T))

    return XPos,SPos,YP,SP,YB
"""
Param = {'nd': 1,
         'nIb': 1,
         'UpdateMode': 2,
         'censor_time': 0.9,
         'xM': np.array([[1, 0], [0, 1]]),
         'dLinkMap': np.array([[0, 1], [1, 1]]),
         'dConstantUpdate': np.array([[0, 1, 1]]),
         'Ak': np.array([[0.9999, 0], [0, 0.9999]]),
         'Bk': np.array([[0, 0], [0, 0]]),
         'Wk': np.array([[0.1107, 0], [0, 0.0607]]),
         'Ck': np.array([[1, 1]]),
         'Dk': np.array([[0, 0, 4.7672]]),
         'Uk': np.array([[0, 0]]),
         'nc': 1,
         'nIn': 3,
         'cLinkMap': np.array([[0, 1], [1, 1]]),
         'cConstantUpdate': np.array([[0, 3, 1]]),
         'Vk': np.array([[0.45]]),
         'S': 0.10,
         'Ek': np.array([[1, 1]]),
         'Fk': np.array([[0, 0, 0]])
         }

compass_filtering(DISTR=[1, 0], tIn=np.array([[1, 0, 1]]), Ib=np.array([[1, 0, 1]]), Yn=0.658, tParam=Param,
                  obs_valid=1,
                  XPos0=np.array([[-4.76676019471412], [0]]), SPos0=np.array([[0.120655723947377, -0.00399920004000000],
                                                                              [-0.00399920004000000,
                                                                               0.0706739399093090]]))
"""