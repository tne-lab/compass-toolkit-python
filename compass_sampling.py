"""
Translation of MATLAB code 'compass_Tk.m' to python
Same as Qk functionally, can therefore be merged together, however for time being
we keep it combined to avoid any confusion
2nd Dec 2022 Translated, Tested & Modified by Sumedh Nagrale
Following modifications done:
1. Removed breaks
2. Removed incorrect bracket syntax
3. Added dummy data for unit testing.
4. Loops tested and confirmed with matlab code
5. removed empty, since it initialized random values.
6. assigned correct initial values
7. renamed the variables
8. Changed the comment from single to double quotes
"""
import numpy as np
import compass_Tk as Ctk
import compass_Qk as Cqk
from scipy.stats import norm
from scipy.stats import gamma


# DISTR,Cut_Time,Uk,In,Ib,Param,XPos0,SPos0
def compass_sampling(DISTR=None,Cut_Time=None, tIn=None, tParam=None, Ib=None, XPos0=None, SPos0=None):
    if DISTR[0] >= 1:
        MCk, MDk = Ctk.compass_Tk(tIn, tParam)
        Ck = tParam['Ck']
        Dk = tParam['Dk'] * MDk  # .* --> * matlab to python for element wise multiplication
        Vk = tParam['Vk']
        if DISTR[0] == 2:
            S = tParam['S']
    if DISTR[1] >= 1:
        MEk, MFk = Cqk.compass_Qk(tIn, tParam)
        Ek = tParam['Ek']
        Fk = tParam['Fk'] * MFk  # .* --> * matlab to python for element wise multiplication
    Ak = tParam['Ak']
    Bk = tParam['Bk']
    Wk = tParam['Wk']
    xM = tParam['xM']
    Uk = tParam['Uk']
    XPre = Ak @ XPos0 + Bk @ Uk.T
    SPre = Ak @ SPos0 @ Ak.T + Wk
    if DISTR[0] == 1:
        CTk = (Ck * MCk[0]) @ xM
        DTk = Dk
        Sk = CTk @ SPre @ CTk.T + Vk
        Yp = CTk @ XPre + DTk @ tIn.T
        # pending on pdf
        print(Yp + 10 * np.sqrt(Sk))
        ys = np.arange(Cut_Time, np.maximum(Cut_Time + 10, Yp + 10 * np.sqrt(Sk)), 0.01)
        Pa = norm.pdf(ys, loc=Yp, scale=np.sqrt(Sk))
        CPa = np.cumsum(Pa)
        CPa = CPa / np.sum(CPa)
        Yn = ys[np.argmin(np.abs(np.random.rand(1) - CPa))]
    if DISTR[0] == 2:
        CTk = (Ck * MCk[0]) @ xM
        DTk = Dk
        EYn = np.exp(CTk @ XPre + DTk @ tIn.T)
        ys = np.arange(Cut_Time, np.maximum(Cut_Time + 10, EYn + (5 * EYn * EYn) / Vk), 0.01)
        ys = ys - S
        Pa = gamma.pdf(ys, a=EYn * Vk, scale=1 / Vk)
        CPa = np.cumsum(Pa)
        CPa = CPa / np.sum(CPa)
        Yn = S + ys[np.argmin(np.abs(np.random.rand(1) - CPa))]
    # needs to be cross checked
    if DISTR[1] == 1:
        ETk = (Ek * MEk[0]) * xM
        FTk = Fk
        temp = ETk @ XPre + FTk @ Ib.T
        pk = np.exp(temp) / (1 + np.exp(temp))
        Yb = 0
        if np.random.rand(1) < pk:
            Yb = 1
    if DISTR[0] == 2 and DISTR[1] == 1:
        return Yb,Yn
    elif DISTR[0] == 2:
        return Yn,np.array([])
    elif DISTR[1] == 1:
        return np.array([]),Yb
"""
Example Input - 
Param = {'nd': 1,
         'nIb': 1,
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
         'Vk': 4.951167805469680,
         'S': 0.10,
         'Ek': np.array([[1,1]]),
         'Fk': np.array([[0,0,0]])
         }
DISTR = [1, 1]

compass_sampling(DISTR=[2, 0], tIn=np.array([[1, 0, 1]]), Ib=np.array([[1,0,1]]), tParam=Param,
                 XPos0=np.array([[-4.76676019471412], [0]]), SPos0=np.array([[0.120655723947377, -0.00399920004000000],
                                                                             [-0.00399920004000000,
                                                                              0.0706739399093090]]), Cut_Time=1)
"""
