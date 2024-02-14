"""Translation of MATLAB code 'compass_set_learning_param.m'

Original Author: Sumedh Sopan Nagrale -- 17 Nov 2022
Translated by Sumedh Sopan Nagrale from MATLAB code to python

Updated:
Author: Sumedh Sopan Nagrale
1) Removing the data argument and keeping it in lines with the compass matlab version,
so that the documentation is easier to maintain.

Author: Sumedh Sopan Nagrale -- 04/01/2024 -- 4th Jan 2024
1) Testing and conforming the validity of the code.
2) The code has been shown to produce same outcome as in case of the MATLAB version
"""


def compass_set_learning_param(Param=None, Iter=None, UpdateStateParam=None,
                               UpdateStateNoise=None, UpdateStateX0=None,
                               UpdateCModelParam=None, UpdateCModelNoise=None,
                               UpdateDModelParam=None, DiagonalA=None,
                               UpdateMode=None, UpdateCModelShift=None):
    """
        Equation:
            X(K+1) = A X(k) + B U(k) + W
            X(0) ~ N(X0,W0)
            Y = CT(k) X(k) + DT(k) + v
            v ~ N(0,sig_v^2)
        Parameters:
        Param (dict): State space parameters
        1: update
        0: do not update
        Iter (int): Number of iteration for EM algorithm training
        UpdateStateParam (int): State transition matrix (A, B) update or not. 1 - true , 0 - false
        UpdateStateNoise (int): W update(1)/not(0)
        UpdateStateX0 (int): mean and covariance matrix
        UpdateCModelParam (int): Continuous observation process (C,D) will be updated or not
        UpdateCModelNoise (int): v i.e. sig_v^2 will be updated or not
        UpdateDModelParam (int): discrete version of UpdateCModelParam
        DiagonalA (int): 1: A is diagonal matrix, 0 A is full matrix
        UpdateMode (int): 1: update mean then covariance 2: update covariance then mean
        UpdateCModelShift (int): 1 shift parameter will be updated, 0 not updated

        Returns:
        Param: A dictionary with information about all the parameters (keys) for state space formulations.
        Additionally, this includes information about what variable needs to be updated during EM algorithm estimations.

        Example: I
        Param = ccss.compass_create_state_space(2, 1, 3, 3, np.eye(2, 2), np.array([1, 2]), np.array([0, 0]), np.array([1, 2]), np.array([0, 0]))
        Iter = 100
        UpdateStateParam = 0
        UpdateStateNoise = 1
        UpdateStateX0 = 1
        UpdateCModelParam = 1
        UpdateCModelNoise = 1
        UpdateDModelParam = 1
        DiagonalA = 1
        UpdateMode = 1
        UpdateCModelShift = 0
        Param = cslp.compass_set_learning_param(Param, Iter, UpdateStateParam,
                                       UpdateStateNoise, UpdateStateX0,
                                       UpdateCModelParam, UpdateCModelNoise,
                                       UpdateDModelParam, DiagonalA,
                                       UpdateMode, UpdateCModelShift)
    """

    """
    Set Learning Rule
    number of iteration
    """
    Param['Iter'] = Iter
    """
    # Hidden model update
    # we can set to 0, which means neither A or B is being updated
    # we can set to 1, which means both A and B are being updated
    # we can set to 2, which means A is being fixed and B is getting updated
    """
    Param['UpdateStateParam'] = UpdateStateParam
    Param['UpdateStateNoise'] = UpdateStateNoise
    Param['UpdateStateX0'] = UpdateStateX0

    ''' continuous model update'''
    Param['UpdateCModelParam'] = UpdateCModelParam
    Param['UpdateCModelNoise'] = UpdateCModelNoise
    '''  discrete model update'''
    Param['UpdateDModelParam'] = UpdateDModelParam
    '''  matrix A'''
    Param['DiagonalA'] = DiagonalA
    ''' Check Update Model'''
    Param['UpdateMode'] = UpdateMode
    '''  Shift in Gamma'''
    Param['UpdateCModelShift'] = UpdateCModelShift

    return Param

