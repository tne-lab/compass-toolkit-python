"""Translation of MATLAB code 'compass_create_state_space.m'

Original Author: Sumedh Sopan Nagrale -- 17 Nov 2022
Translated by Sumedh Sopan Nagrale from MATLAB code to python

Updated:
Author: Sumedh Sopan Nagrale
1) Removing the data argument and keeping it in lines with the compass matlab version,
so that the documentation is easier to maintain.

Author: Sumedh Sopan Nagrale -- 04/01/2024 -- 4th Jan 2024
1) Testing and conforming the validity of the code.
2) The code has been shown to produce same outcome as in case of the MATLAB version

Author: Sumedh Sopan Nagrale -- 08/01/2024 -- 8th Jan 2024
1) Editing for the index, of clinkupdate
"""
import numpy as np


def compass_create_state_space(nx=None, nUk=None, nIn=None, nIb=None, xM=None, cLink=None, cLinkUpdate=None, dLink=None,
                               dLinkUpdate=None):
    """
    This function construct a dictionary (bc it's fast) that includes information of the overall setup.
    Equation:
            X(K+1) = A X(k) + B U(k) + W
            X(0) ~ N(X0,W0)
            Y = CT(k) X(k) + DT(k) + v
            v ~ N(0,sig_v^2)
    Parameters:
    nx (int): Number of states for the state space equations. (in our context, x_baseline and x_conflict)
    nUk (int): Number of inputs to the continuous part of the model.
    nIn (int): Indicator function.
    nIb (int): Number of inputs to the discrete part of the model.
    xM (int): relate the states with the observation process (for both discrete and continuous)
    cLink (numpy array): here we consider tx states (tx < nx), Clink provides links the column of In to the states.
    cLinkUpdate (numpy array): Determines which of the parameter associated needs to be updated.
    dLink (numpy array): same as cLink but for discrete.
    dLinkUpdate (numpy array): same as cLinkUpdate but for discrete.

    Returns:
    Param: A dictionary with information about all the parameters (keys).

    Example: I
        nx = 2
        nUk = 12
        nIn = 3
        nIb = 0  # need to note this
        # Reason note: This is assigned a number rather than empty bc the differences between the
        xM = np.eye(2, 2)
        cLink = np.array([1, 2])
        cLinkUpdate = np.array([0, 0])
        dLink = np.array([])
        dLinkUpdate = np.array([])
        print(compass_create_state_space(nx, nUk, nIn, nIb, xM, cLink, cLinkUpdate, dLink, dLinkUpdate))

    Example: II
    # MATLAB code
    Param = compass_create_state_space(1,1,2,2,eye(1,1),1,1,1,0);
    # Python code
    compass_create_state_space(1, 1, 2, 2, np.eye(1, 1), np.array([1]), np.array([1]), np.array([1]), np.array([0]))

    Example: III
    # MATLAB code
    Param = compass_create_state_space(2,1,3,3,eye(2,2),[1 2],[0 0],[1 2],[0 0]);
    # Python code
    compass_create_state_space(2, 1, 3, 3, np.eye(2, 2), np.array([1, 2]), np.array([0, 0]), np.array([1, 2]), np.array([0, 0]))
    """

    """Build The State-Space Model
    """
    nc = 1
    nd = 1
    # don't need to get Param, it will be formed here
    Param = {}
    '''Dimension of Components'''
    '''Size of X state vector'''
    Param['nx'] = nx
    Param['nc'] = nc
    Param['nd'] = nd

    '''Input and Their Link Functions'''
    '''Input and its link to the continuous model'''
    Param['nIn'] = nIn  # Bias, I, V, I2C, C2I

    if cLink.size != 0 and cLinkUpdate.size != 0:
        '''The cLinkMap - continuous link map'''
        # this does not make sense to have a array of array
        Param['cLinkMap'] = np.zeros((nc, xM.shape[0]))
        Param['cLinkUpdate'] = np.zeros((nc, xM.shape[0]))
        print(np.arange(0, nc, 1))
        for i in range(0, nc):
            Param['cLinkMap'][i, :] = cLink[i:]
            Param['cLinkUpdate'][i, :] = cLinkUpdate[i:]

    ''' Input and its link to the discrete model'''
    Param['nIb'] = nIb  # Bias, I, V, I2C, C2I
    if dLink.size != 0 and dLinkUpdate.size != 0:
        '''The dLinkMap - discrete link map'''
        Param['dLinkMap'] = np.zeros((nd, xM.shape[0]))
        Param['dLinkUpdate'] = np.zeros((nd, xM.shape[0]))
        for i in range(0, nd):
            Param['dLinkMap'][i, :] = dLink[i:]
            Param['dLinkUpdate'][i, :] = dLinkUpdate[i:]

    '''Model parameters'''
    '''Hidden state model'''
    Param['Ak'] = np.eye(nx, nx) * 1
    Param['Bk'] = np.eye(nx, nUk) * 0.0  # INPUT
    Param['Wk'] = np.eye(nx, nx) * 0.05  # IID NOISE
    Param['XO'] = np.zeros((nx, 1))  # initial x0 is set to 0
    Param['WO'] = np.eye(nx, nx) * 1  # initial x0 is set to 0

    '''Continuous model'''
    Param['Ck'] = np.ones((nc, xM.shape[0]))  # Coefficients of the x
    Param['Dk'] = np.ones((nc, nIn))  # Input parameters

    '''we need to drop some input from update'''
    """flawed logic, the logic depends on the language not a clear difference between logic and making use of the underlying structure"""
    Param['cConstantUpdate'] = np.ones((nc, nIn))
    if cLink.size != 0 and cLinkUpdate.size != 0:
        for i in range(0, nc):
            ind = np.argwhere(Param['cLinkMap'][i, :])
            if ind.size > 0:
                cInd = (Param['cLinkMap'][i, ind]).ravel().astype(int) - 1 # adjusting the index
                Param['cConstantUpdate'][i, cInd] = 0

    Param['Vk'] = np.eye(nc, nc) * 0.01  # noise

    # Discrete model
    Param['Ek'] = np.ones((nd, np.size(xM, 1)))  # coefficient of the x
    Param['Fk'] = np.ones((nd, Param['nIb']))  # input parameters

    '''we need to drop some input from update'''
    if dLink.size != 0 and dLinkUpdate.size != 0:
        Param['dConstantUpdate'] = np.ones((nd, Param['nIb']))
        for i in range(0, nd):
            ind = np.argwhere(Param['dLinkMap'][i, :])
            if ind.size > 0:
                dInd = ((Param['dLinkMap'][i, ind])).ravel().astype(int) - 1
                Param['dConstantUpdate'][i, dInd] = 0
    ''' xM is the X Mapping'''
    Param['xM'] = xM

    ''' two extra parameters for Gamma (Param.Vk is being treated as dispersion)'''
    Param['S'] = 0

    ''' set censor_time + processing model'''
    Param['censor_time'] = 1
    Param['censor_mode'] = 1
    ''' mode 1: sampling, mode 2: full likelihood'''
    # mode 1 is applicable on all data types
    # mode 2 is defined on continuous variable

    Param['censor_update_mode'] = 1  # how the run the filter rule

    ''' it define how the Kalman Filter is getting updated - two different methods'''
    Param['UpdateMode'] = 1

    return Param
