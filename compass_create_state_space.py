"""Translation of MATLAB code 'compass_create_state_space.m' to python 17 Nov 2022  Translated by Sumedh Nagrale
1)removing the data argument and keeping it in lines with the compass matlab version, so that the documentation is
easier to maintain.
"""

import numpy as np

def compass_create_state_space(nx, nUk, nIn, nIb, xM, cLink, cLinkUpdate, dLink, dLinkUpdate):
    """Build The State-Space Model"""
    nc = 1
    nd = 1
    # don't need to get Param, it will be formed here
    Param = {}
    '''Dimension of Components'''
    '''Size of X vector'''
    Param['nx'] = nx
    Param['nc'] = 1
    Param['nd'] = 1

    '''Input and Their Link Functions'''
    '''Input and its link to the continuous model'''
    Param['nIn'] = nIn  # Bias, I, V, I2C, C2I

    if cLink.size != 0 and cLinkUpdate.size != 0:
        # not cLink[0] and not cLinkUpdate[0]:
        '''The cLinkMap - continuous link map'''
        Param['cLinkMap'] = np.zeros((nc, xM.shape[0]))
        Param['cLinkUpdate'] = np.zeros((nc, xM.shape[0]))
        for i in range(0, nc):
            Param['cLinkMap'][i, :] = cLink[i, :]
            Param['cLinkUpdate'][i, :] = cLinkUpdate[i, :]

    ''' Input and its link to the discrete model'''
    Param['nIb'] = nIb  # Bias, I, V, I2C, C2I
    if dLink.size != 0 and dLinkUpdate.size != 0:
        '''The dLinkMap - discrete link map'''
        Param['dLinkMap'] = np.zeros((nd, xM.shape[0]))
        Param['dLinkUpdate'] = np.zeros((nd, xM.shape[0]))
        for i in range(0, nd):
            Param['dLinkMap'][i, :] = dLink[i]
            Param['dLinkUpdate'][i, :] = dLinkUpdate[i]

    '''Model parameters'''
    '''Hidden state model'''
    # removing the Param from here, since no update above and it's easier nx to read
    Param['Ak'] = np.eye(nx, nx) * 1
    Param['Bk'] = np.eye(nx, nUk) * 0.0  # INPUT
    Param['Wk'] = np.eye(nx, nx) * 0.05  # IID NOISE
    Param['XO'] = np.zeros((nx, 1))  # initial x0 is set to 0
    Param['WO'] = np.eye(nx, nx) * 1  # initial x0 is set to 0

    '''Continuous model'''
    Param['Ck'] = np.ones((nc, xM.shape[0]))  # Coefficients of the x
    Param['Dk'] = np.eye(nc, nIn)  # Input parameters

    '''we need to drop some input from update'''
    Param['cConstantUpdate'] = np.ones((nc, nIn))
    if cLink.size != 0 and cLinkUpdate.size != 0:
        # not cLink[0] and not cLinkUpdate[0]:
        for i in range(0, nc):
            ind = np.argwhere(Param['cLinkMap'][i, :])
            if ind.size > 0:
                # Param['cConstantUpdate'][i, Param['cLinkMap'][i, ind]] = 0
                Param['cConstantUpdate'][i, ind] = 0

    Param['Vk'] = np.eye(nc, nc) * 0.01  # noise

    # Discrete model
    Param['Ek'] = np.ones((nd, np.size(xM, 1)))  # coefficient of the x
    Param['Fk'] = np.ones((nd, Param['nIb']))  # input parameters

    '''we need to drop some input from update'''
    if dLink.size != 0 and dLinkUpdate.size != 0:
        # not dLink[0] and not dLinkUpdate[0]:
        Param['dConstantUpdate'] = np.ones((nd, Param['nIb']))
        for i in range(0, nd):
            ind = np.argwhere(Param['dLinkMap'][i, :])
            if ind.size > 0:
                # Param['dConstantUpdate'][i, Param['dLinkMap'][i, ind]] = 0
                Param['dConstantUpdate'][i, ind] = 0
    ''' xM is the X Mapping'''
    Param['xM'] = xM

    ''' two extra paramaters for Gamma (Param.Vk is beging treated as Dispression)'''
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
