"""
Translation of MATLAB code 'compass_post_sampling.m' to python
16th September 2023  Translated, Tested & Modified by Sumedh Nagrale
Following modifications were made to
1. Removing bugs
"""
import compass_Tk as Ctk
import numpy as np
import compass_Qk as Cqk
from scipy.stats import norm
from scipy.stats import gamma


def compass_post_sampling(DISTR=None, Cut_Time=None, In=None, Ib=None, Param=None, XSmt=None, SSmt=None):
    ''' Input Argument
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
    % Param: it keeps the model information, and paramaters '''
    ''' Output Argument
    % YP reaction time
    % YB binary decision'''
    '''Build Mask Ck, Dk ,EK and Fk - note that Ck, Ek are time dependent and the Dk and Fk is linked to a subset of 
    Input '''
    Yn = []
    Yb = []
    if DISTR(0) >= 1:
        [MCk, MDk] = Ctk.compass_Tk(In, Param)
        '''%% Normal Observation Model (  Y(k)=(Ck.*Tk)*X(k)+Dk*Ik+Vk*iid white noise )
        % ------------------
        % Y(k) is the observation, and Ik is the input either indicator or continuous
        % Ck, Dk, Vk are the model paramateres
        % Tk is model specific function - it is original set to but a one matrix
        % ------------------
        % Ck, NxM matrix - (Y is an observation of the length N, N can be 1, 2, ... - The Tk has the same size of input, 
        % and it is specfically designed for our task. It can be set to all 1 matrix)'''
        Ck = Param['Ck']
        '''Bk, NxS3 matrix - (We have an input of the length S3, and Dk will be size of NxS3)'''
        Dk = Param['Dk'] * MDk
        '''Vk, is NxS4 matrix - (This is observation noise; for this, the S4 will be generally equal to N)'''
        Vk = Param.Vk
        '''S'''
        if DISTR[0] == 2:
            S = Param['S']

    if DISTR(1) >= 1:
        [MEk, MFk] = Cqk.compass_Qk(Ib, Param)
        '''%% Binary Observation Model (  P(k)=sigmoid((Ek.*Qk)*X(k)+Fk*Ik) )
        % ------------------
        % P(k) is the observation probability at time k, and Ik is the input either indicator or continuous
        % Ek, and Fk are the model paramateres
        % Qk is model specific function - it is original set to but a one matrix
        % ------------------
        % Ck, NxM matrix - similar to Ck, Tk'''
        Ek = Param['Ek']
        '''Fk, NxS5 matrix - Similar to Dk'''
        Fk = Param['Fk'] * MFk

    '''State Space Model (  X(k+1)=Ak*X(k)+Bk*Uk+Wk*iid white noise ) xMapping'''
    xM = Param['xM']
    '''Data observation: Normal'''
    if DISTR(0) == 1:  # main observation is Normal
        # Filtering
        CTk = (Ck @ MCk[0]) @ xM
        DTk = Dk
        # YP, Sk
        Sk = CTk @ SSmt @ CTk.T + Vk
        Yp = CTk @ XSmt + DTk @ In.T
        # Generate a sample - we assume it is scalar
        if not Cut_Time:
            Yn = Yp + np.sqrt(Sk) @ np.randn()
        else:
            ys = np.arange(Cut_Time, np.maximum(Cut_Time + 10, Yp + 10 * np.sqrt(Sk)), 0.01)
            Pa = norm.pdf(ys, loc=Yp, scale=np.sqrt(Sk))
            CPa = np.cumsum(Pa)
            CPa = CPa / CPa[-1]
            ui = np.argmin(np.abs(np.random.rand(1)- CPa))
            Yn = ys(ui)
    if DISTR(0) == 2:  # main observation is Gamma
        # Filtering
        CTk = (Ck @ MCk[0]) @ xM
        DTk = Dk
        # YP, Sk
        EYn = np.exp(CTk @ XSmt + DTk @ In.T)
        if not Cut_Time:
            Yn = S + np.gamrnd(Vk, EYn / Vk)
        else:
            ys = np.arange(Cut_Time, np.maximum(Cut_Time + 10, EYn + (5 * EYn * EYn) * 1/Vk), 0.01)
            ys = ys - S
            Pa = gamma.pdf(ys, a=EYn @ Vk, scale=np.linalg.inv(Vk))
            CPa = np.cumsum(Pa)
            CPa = CPa / CPa[-1]
            ui = np.argmin(np.abs(np.random.rand(1) - CPa))
            Yn = S + ys(ui)

    if DISTR(1) == 1:
        ETk = (Ek * MEk[0]) @ xM
        FTk = Fk
        # calculate p
        temp = ETk @ XSmt + FTk @ Ib.T
        pk = np.divide(np.exp(temp), (1 + np.exp(temp)))
        Yb = 0
        if np.rand() < pk:
            Yb = 1

    # return variables
    if DISTR[0] == 2 and DISTR[1] == 1:
        return Yb, Yn
    elif DISTR[0] == 2:
        return Yn, np.array([])
    elif DISTR[1] == 1:
        return np.array([]), Yb