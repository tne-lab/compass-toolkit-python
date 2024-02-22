"""Translation of MATLAB code 'compass_set_censor_threshold_proc_mode.m'

Original Author: Sumedh Sopan Nagrale -- 04/01/2024 -- 4th Jan 2024
Translated by Sumedh Sopan Nagrale from MATLAB code to python
"""


def compass_set_censor_threshold_proc_mode(Param=None, censor_thr=None, censor_mode=None, update_mode=None):

    """
       Equation:
           X(K+1) = A X(k) + B U(k) + W
           X(0) ~ N(X0,W0)
           Y = CT(k) X(k) + DT(k) + v
           v ~ N(0,sig_v^2)

       Parameters:
       censor_thr (int) : censor time for observation process
       censor_mode (int) : censor method, 1: single imputation and 2: Gaussian process-full-likelihood method
       update_mode (int) :`1: update mean then covariance 2: update covariance then mean

       Returns:
       Param: A dictionary with information about all the parameters (keys) for state space formulations, EM algorithm
       updates.
       Additionally, this includes information about what kind of censoring method needs to be used.

       Example: I
        censor_thr = np.log(2)
        censor_mode = 1
        update_mode = 1
        Param = compass_set_censor_threshold_proc_mode(Param, censor_thr, censor_mode, update_mode)

        MATLAB:
        Param = compass_set_censor_threshold_proc_mode(Param,log(2),1,1);

    """
    Param["censor_time"] = censor_thr
    Param["censor_mode"] = censor_mode
    '''
    mode 1: sampling, 
    mode 2: full likelihood
    '''
    Param["censor_update_mode"] = update_mode
    '''
    either 1 or 2 
    mode 2 is valid when there is a continuous variable
    '''

    return Param
