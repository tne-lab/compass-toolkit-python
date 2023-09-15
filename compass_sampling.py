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

15th Sep 2023
Following modifications were made to
1. Removing bugs
"""
import numpy as np
import compass_Tk as Ctk
import compass_Qk as Cqk
from scipy.stats import norm
from scipy.stats import gamma


# DISTR,Cut_Time,Uk,In,Ib,Param,XPos0,SPos0
def compass_sampling(DISTR=None, Cut_Time=None, tIn=None, tParam=None, Ib=None, XPos0=None, SPos0=None):
    """
    %% Input Argument
    % DISTR, a vecotr of two variables. The [1 0] means there is only normal
    % observation/s, [0 1] means there is only binary observation/s, and [1 1]
    % will be both observations.
    % Uk: is a matrix of size KxS1 - K is the length of observation - input to
    % State model - X(k)=A*X(k-1)+B*Uk+Wk
    % In: is a matrix of size KxS3 - K is the length of observation - input to Normal observation model
    % Yn(k)=(C.*MCk)*X(k)+(D.*MDk)*In+Vk       - C and D are free parameters,
    % and MCk and MDk are input dependent components
    % Ib: is a matrix of size KxS5 - K is the length of observation - input to Binary observation model
    %  P(Yb(k)==1)=sigmoid((E.*MEk)*X(k)+(F.*MFk)*Ib       - E and F are free parameters,
    % and MEk and MFk are input dependent components
    % Yn: is a matrix of size KxN  - K is the length of observation, matrix of
    % normal observation
    % Yb: is a matrix of size KxN  - K is the length of observation, matrix of
    % binary observation
    % Param: it keeps the model information, and paramaters
    %% Output Argument
    % YP reaction time
    % YB binary decision
    """
    # Build Mask Ck, Dk ,EK and Fk - note that Ck, Ek are time dependent and the Dk and Fk is linked to a subset of
    # Input
    Yn = []
    Yb = []
    if DISTR[0] >= 1:
        [MCk, MDk] = Ctk.compass_Tk(tIn, tParam)
        """
        %% Normal Observation Model (  Y(k)=(Ck.*Tk)*X(k)+Dk*Ik+Vk*iid white noise )
        % ------------------
        % Y(k) is the observation, and Ik is the input either indicator or continuous
        % Ck, Dk, Vk are the model paramateres
        % Tk is model specific function - it is original set to but a one matrix
        % ------------------
        % Ck, NxM matrix - (Y is an observation of the length N, N can be 1, 2, ... - The Tk has the same size of input, 
        % and it is specfically designed for our task. It can be set to all 1 matrix)
        """
        Ck = tParam['Ck']
        # Bk, NxS3 matrix - (We have an input of the length S3, and Dk will be size of NxS3)
        Dk = tParam['Dk'] * MDk  # .* --> * matlab to python for element wise multiplication
        # Vk, is NxS4 matrix - (This is observation noise; for this, the S4 will be generally equal to N)
        Vk = tParam['Vk']
        # S
        if DISTR[0] == 2:
            S = tParam['S']

    if DISTR[1] >= 1:
        MEk, MFk = Cqk.compass_Qk(tIn, tParam)
        """
        %% Binary Observation Model (  P(k)=sigmoid((Ek.*Qk)*X(k)+Fk*Ik) )
        % ------------------
        % P(k) is the observation probability at time k, and Ik is the input either indicator or continuous
        % Ek, and Fk are the model paramaters
        % Qk is model specific function - it is original set to but a one matrix
        % ------------------
        % Ck, NxM matrix - similar to Ck, Tk
        """
        Ek = tParam['Ek']
        # Fk, NxS5 matrix - Similar to Dk
        Fk = tParam['Fk'] * MFk  # .* --> * matlab to python for element wise multiplication

    """
    %% State Space Model (  X(k+1)=Ak*X(k)+Bk*Uk+Wk*iid white noise )
    % ------------------
    % X(k) is the state, and Uk is the input
    % Ak, Bk, Wk are model paramateres
    % ------------------
    % Ak, MxM matrix  (M is the length of the X)
    """
    Ak = tParam['Ak']
    # Bk, MxS1 matrix (S1 is the length of Uk, Uk is a vector of size S1x1)
    Bk = tParam['Bk']
    # Wk, is MxS2 matrix (S2 is the length of Noise, we normally set the noise with the same dimension as the X - S2=M)
    Wk = tParam['Wk']
    # xMapping
    xM = tParam['xM']
    """Filtering Section"""
    Uk = tParam['Uk']
    XPre = Ak @ XPos0 + Bk @ Uk.T
    SPre = Ak @ SPos0 @ Ak.T + Wk

    """Data observation: Normal"""
    if DISTR[0] == 1:  # main observation is Normal
        #  Filtering
        CTk = (Ck * MCk[0]) @ xM
        DTk = Dk
        # YP, Sk
        Sk = CTk @ SPre @ CTk.T + Vk
        Yp = CTk @ XPre + DTk @ tIn.T
        # Generate a sample - we assume it is scalar
        ys = np.arange(Cut_Time, np.maximum(Cut_Time + 10, Yp + 10 * np.sqrt(Sk)), 0.01)
        Pa = norm.pdf(ys, loc=Yp, scale=np.sqrt(Sk))
        CPa = np.cumsum(Pa)
        CPa = CPa / np.sum(CPa)
        Yn = ys[np.argmin(np.abs(np.random.rand(1) - CPa))]
    """main observation is Gamma"""
    if DISTR[0] == 2:
        CTk = (Ck * MCk[0]) @ xM
        DTk = Dk
        # YP, Sk
        EYn = np.exp(CTk @ XPre + DTk @ tIn.T)
        # Generate a sample - we assume it is scalar
        ys = np.arange(Cut_Time, np.maximum(Cut_Time + 10, EYn + (5 * EYn * EYn) * 1/Vk), 0.01)
        ys = ys - S
        Pa = gamma.pdf(ys, a=EYn @ Vk, scale=np.linalg.inv(Vk))
        CPa = np.cumsum(Pa)
        CPa = CPa / np.sum(CPa)
        Yn = S + ys[np.argmin(np.abs(np.random.rand(1) - CPa))]

    if DISTR[1] == 1:
        ETk = (Ek * MEk[0]) * xM
        FTk = Fk
        # calculate p
        temp = ETk @ XPre + FTk @ Ib.T
        pk = np.exp(temp) / (1 + np.exp(temp))
        Yb = 0
        if np.random.rand(1) < pk:
            Yb = 1
    # return variables
    if DISTR[0] == 2 and DISTR[1] == 1:
        return Yb, Yn
    elif DISTR[0] == 2:
        return Yn, np.array([])
    elif DISTR[1] == 1:
        return np.array([]), Yb


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
         'Ek': np.array([[1, 1]]),
         'Fk': np.array([[0, 0, 0]])
         }
DISTR = [1, 1]

print(compass_sampling(DISTR=[2, 0], tIn=np.array([[1, 0, 1]]), Ib=np.array([[1, 0, 1]]), tParam=Param,
                       XPos0=np.array([[-4.76676019471412], [0]]),
                       SPos0=np.array([[0.120655723947377, -0.00399920004000000],
                                       [-0.00399920004000000,
                                        0.0706739399093090]]), Cut_Time=1))
