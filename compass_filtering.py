'''
Translated MATLAB code 'compass_filtering.m' to python
17 Nov 2022 - Jazlin Taylor
'''
import numpy as np
import compass_Qk
import compass_Tk
import compass_sampling

def compass_filtering(DISTR, Uk, In, Ib, Yn, Yb, Param, obs_valid, XPos0, SPos0):
    '''
    ## Input Arguments
        DISTR - a vecotr of two variables. The [1 0] means there is only normal observation/s,
            [0 1] means there is only binary observation/s, and [1 1] will be both observations.

        Uk - matrix of size K x S1 where K is length of observation, Uk is input to state model 
            of X=A*X(k-1)+B*Uk+Wk

        In - matrix of size K x S3, where K is length of observation, In is input to Normal observation 
             model of Yn = (C.*MCk)*X(k)+(D.*MDk)*In+Vk where C and D are free parameters, MCk and MDk 
             are input dependent components
        
        Ib - (matrix of size K x S5, where K is length of observation) input to Binary observation model 
             of P(Yb==1)=sigmoid((E.*MEk)*X(k)+(F.*MFk)*Ib where E and F are free parameters, MEk and MFk 
             are input dependent components

        Yn - matrix of normal observation of size K x N

        Yb - matrix of binary observation of size K x N

        Param - keeps model information and parameters

    ## Output Arguments
        Xsmt - mean smoothing result
        SSmt - variance smoothing result
        Param - updated model parameters
        XPos - mean filtering result
        SPos - variance filtering result
        ML - value of E-step maximization
        YP - prediction of Yn
        YB is not added yet but can be prediciton of binary probability
    '''

    update_mode = Param['UpdateMode'] #assuming coming from dictionary

    ## Observation Mode, from 1 to 5
    if DISTR(0) == 1:
        observe_mode = DISTR(0) + 2*DISTR(1)
    elif DISTR(0) == 2:
        observe_mode = 2*DISTR(0) + DISTR(1)
    else:
        observe_mode = 2*DISTR(1)
    
    ## Build Mask Ck, Dk ,EK and Fk - note that Ck, Ek are time dependent
    # and the Dk and Fk is linked to a subset of Input
    MCk, MDk = compass_Tk(In, Param)
    if DISTR(1) == 1:
        MEk, MFk = compass_Qk(Ib, Param)
    
    ## State Space Model (X(k+1)= Ak*X(k) + Bk*Uk + Wk * iid white noise)
    '''
    X(k) is the state, and Uk is the input
    Ak, Bk, Wk are model paramateres
    '''
    # Ak is MxM matrix (M is length of the X)
    Ak = Param['Ak']

    # Bk is MxS1 matrix (S1 is length of Uk, Uk is vector of size S1x1)
    Bk = Param['Bk']

    # Wk is MxS2 matrix (S2 is length of noise, we normally set the noise as same dim. as the X - S2 = M)
    Wk = Param['Wk']
    xM = Param['xM'] #This extends x

    ## Censored reaction time
    if not np.argwhere(obs_valid == 2): #not sure about this one
        censor_time = Param['censor_time']
        
    ## Normal/Gamma Observation Model
    if DISTR(0) > 0:
        '''
    For Normal,  Y(k)=(Ck.*Tk)*X(k)+Dk*Ik + Vk    Vk variance of iid white noise
    For Gamma,   Y(k)=exp((Ck.*Tk)*X(k)+Dk*Ik)    Vk is dispersion term 
    ------------------
    Y(k) is the observation, and Ik is the input either indicator or continuous
    Ck, Dk, Vk are the model paramateres
    Tk is model specific function - it is original set to but a one matrix
    ------------------
        ''' 
        # Ck (1xM matrix) - (Y is an scalar observation at each time point ... - The Tk has the same size of input,
        # and it is specfically designed for our task. It can be set to all 1 matrix)
        Ck = Param['Ck']

        # Bk (NxS3 matrix) - We have an input of length S3 so Dk will be of size NxS3
        Dk = np.multiply(Param['Dk'], MDk) #matlab is .*

        #Vk is a scalar representing noise in Normal or dispersion term in Gamma
        Vk = Param['Vk']

    ## Binary Observation Model (P(k)=sigmoid((Ek.*Qk)*X(k)+(Fk*Ik))
    if DISTR(1) == 1:
        '''
    ------------------
    P(k) is the observation probability at time k, and Ik is the input either indicator or continuous
    Ek, and Fk are the model paramateres
    Qk is model specific function - it is original set to but a one matrix
    ------------------
        '''
        # Ek (NxM matrix) similar to Ck, Tk
        Ek = Param['Ek']

        # Fk (NxS5 matrix) similar to Dk
        Fk = np.multiply(Param['Fk'], MFk)

    ## Check Uk
    if not Uk:
        Uk = np.empty((1, np.size(Bk, 1)))

    ## Filter
    # One step prediction

    XPre = Ak * XPos0 + Bk * np.transpose(Uk)
    SPre = Ak * SPos0 * np.transpose(Ak) + Wk

    # Check if data point is censored
    if obs_valid >= 1:
        # Draw sample if its censored
        if obs_valid == 2:
            tYP, tYB = compass_sampling(DISTR, censor_time, Uk, In, Ib, Param, XPre, SPre)
            if DISTR(0) > 0:
                Yn = tYP
            elif DISTR(1) == 1:
                Yb = tYB
        
        # Observation: Normal
        if observe_mode == 1:
            CTk = np.multiply(Ck, MCk[0]) * xM
            DTk = Dk
            
            # XPos
            Sk =  CTk * SPre * np.transpose(CTk) + Vk
            Yp =  CTk * XPre + DTk * np.transpose(In)
            XPos =  XPre + SPre * np.transpose(CTk) * Sk**-1* (Yn-Yp) #not sure what ^-1 is supposed to be, inverse?
            
            # SPos
            SPos = (SPre**-1 + np.transpose(CTk) * Vk**-1 * CTk)**-1 #same here

        # Observation: Bernoulli
        if observe_mode == 2:
            ETk = np.multiply(Ek, MEk) * xM
            FTk = Fk

            # XPos, SPos
            # Recursive mode
            if update_mode == 1:
                in_loop = 10
                # XPos update
                xpos = XPre
                for t in range(0, in_loop):
                    st = ETk * xpos + FTk * np.transpose(Ib)
                    pk = np.divide(np.exp(st), (1+np.exp(st)))
                    xpos = XPre + SPre * np.transpose(ETk) * (Yb - pk)
                    break
                XPos = xpos
                # SPos update
                SPos = (SPre**-1 + np.transpose(ETk) * np.diag(np.multiply(pk, 1-pk))*ETk)**-1
            
            # One step mode
            if update_mode == 2:
                st = ETk * XPre + FTk * np.transpose(Ib)
                pk = np.divide(np.exp(st),1+np.exp(st))
                SPos = (SPre**-1 + np.transpose(ETk) * np.diag(np.multiply(pk, 1-pk))*ETk)**-1
                XPos = XPre + SPos * np.transpose(ETk) * (Yb - pk)

        # Observation: Normal + Bernoulli
        if observe_mode == 3:
            CTk = np.multiply(Ck, MCk[0]) * xM
            DTk = Dk
            ETk = np.multiply(Ek, MEk[0]) * xM
            FTk = Fk

            # XPos, SPos
            # Recursive mode
            if update_mode == 1:
                # XPos update 
                in_loop = 10
                xpos = XPre
                Yp = CTk * XPre + DTk * np.transpose(In)
                Sk = np.transpose(CTk) * Vk**-1 * CTk + SPre**-1
                for t in range(0,in_loop):
                    st = ETk * xpos + FTk * np.transpose(Ib)
                    pk = np.divide(np.exp(st),1+np.exp(st))
                    xpos = XPre +  Sk**-1 * (np.transpose(ETk) *(Yb-pk) + np.transpose(CTk)* Vk**-1 *(Yn-Yp))
                    break
                XPos = xpos
                # SPos Update
                SPos = (SPre**-1 + np.transpose(CTk) * Vk**-1 * CTk + np.transpose(ETk) * np.diag(np.multiply(pk, 1-pk))*ETk)**-1
            
            # One step mode
            if update_mode == 2:
                Yp   =  CTk * XPre + DTk * np.transpose(In)
                st   =  ETk * XPre + FTk * np.transpose(Ib)
                pk   =  np.divide(np.exp(st),1+np.exp(st))
                SPos = (SPre**-1 + np.transpose(ETk) * np.diag(np.multiply(pk, (1-pk)))*ETk + np.transpose(CTk) * Vk**-1 * CTk )**-1
                XPos = XPre +  SPos * (np.transpose(ETk) * (Yb-pk) + np.transpose(CTk) * (Yn-Yp) * Vk**-1)

        # Observation: Gamma
        if observe_mode == 4:
            CTk = np.multiply(Ck, MCk[0]) * xM #not sure if indexing this or something else
            DTk = Dk

            # XPos, SPos
            # Recursive mode
            if update_mode == 1:
                # XPos update
                Yk = Yn - Param['S']
                in_loop = 10
                xpos = XPre
                for t in range(0, in_loop):
                    Yp = np.exp(CTk * xpos + DTk * np.transpose(In))
                    xpos = XPre - SPre * Vk * np.transpose(CTk) * (1-Yk/Yp) #check order of operations
                    break
                XPos = xpos
                SPos = (SPre**-1 + (Vk*(Yk/Yp)) * np.transpose(CTk) * CTk)**-1 #Order of operations????
            
            # One step mode
            if update_mode == 2:
                Yk = Yn - Param['S']
                Yp = np.exp(CTk * XPre + DTk * np.transpose(In))
                SPos = (SPre**-1 + (Vk*(Yk/Yp)) * np.transpose(CTk) * CTk)**-1
                XPos = XPre - SPos * Vk * np.transpose(CTk) * (1-Yk/Yp)
        
        # Observation: Gamma + Bernoulli
        if observe_mode == 5:
            CTk = np.multiply(Ck, MCk[0]) * xM
            DTk = Dk
            ETk = np.multiply(Ek, MEk[0]) * xM
            FTk = Fk

            # Recursive mode
            if update_mode == 1:
                # XPos update
                in_loop = 10
                Yk = Yn - Param['S']
                xpos = XPre
                for t in range(0, in_loop):
                    st = ETk * xpos + FTk * np.transpose(Ib)
                    pk = np.divide(np.exp(st), 1+np.exp(st))
                    Yp = np.exp(CTk * xpos + DTk * np.transpose(In))
                    xpos = XPre + SPre * (np.transpose(ETk) * (Yb - pk) - Vk * np.transpose(CTk) * (1-Yk/Yp))
                    break
                XPos = xpos

                # SPos update
                SPos = (SPre**-1 + np.transpose(CTk) * CTk * Vk * (Yk/Yp) + np.tranpose(ETk) * ETk * np.diag(np.multiply(pk, 1-pk)))**-1

            # One step mode
            if update_mode == 2:
                Yk = Yn - Param['S']
                Yp = np.exp(CTk * XPre + DTk * In.T)
                st = ETk * XPre + FTk * np.transpose(Ib)
                pk = np.divide(np.exp(st), 1 + np.exp(st))
                SPos = (SPre**-1 + np.transpose(CTk) * CTk * Vk * (Yk/Yp) + np.tranpose(ETk) * ETk * np.diag(np.multiply(pk, 1-pk)))**-1
                XPos = XPre + SPre * (np.transpose(ETk) * (Yb - pk) - Vk * np.transpose(CTk) * (1-Yk/Yp))

    else:
        # Randomly censored, the filter estimate will be equal to one step prediction
        XPos = XPre
        SPos = SPre

    
    ## Update prediction
    YP = ()
    YB = ()

    if DISTR(0) > 0:
        #Filtering
        CTk = np.multiply(Ck, MCk[0]) * xM
        DTk = Dk
        #EYn
        if DISTR(0) == 1:
            temp = CTk * XPos + DTk * np.transpose(In)
            YP = np.transpose(temp)
        else:
            temp = CTk * XPos + DTk * np.transpose(In)
            YP = np.exp(temp) * np.exp(0.5 * CTk * SPos * np.transpose(CTk))
            SP = np.exp(2*temp) * np.exp(2 * CTk * SPos * np.transpose(CTk)) - YP*YP
            YP = Param['S'] + YP
    
    if DISTR(1) == 1:
        #Filtering
        ETk = np.multiply(Ek, MEk[0]) * xM
        FTk = Fk
        #YP
        temp = ETk * XPos + FTk * np.transpose(Ib)
        YB = np.divide(np.exp(np.transpose(temp)), 1+np.exp(np.transpose(temp)))
    
    #returnn
    return XPos, SPos, YP, SP, YB



