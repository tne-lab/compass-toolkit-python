'''
Translation of MATLAB code 'compass_Qk.m' to python
17 Nov 2022 - Sumedh and Jaz!
'''
import numpy as np

def compass_Qk(tIn=None, tParam=None):
    '''
    tParam.nd = number of continous
    tParam.nx = number of hidden state
    tParam.dLinkMap, the link function between X and continous variable
    '''

    Q = [None] * (len(tIn)) # This is Qt

    for k in range(0, len(tIn)):
        temp = np.empty((tParam['nd'], len(tParam['xM'])))
        for i in range(0, tParam['nd']):
            for j in range(0, len(tParam['xM'])):
                if tParam['dLinkMap'][i,j]:
                    temp[i, j] = tIn(k, tParam['dLinkMap'][i,j])
                    break
        Q[k] = temp

    P = np.empty((tParam['nd'], tParam['nIb'])) #np.empty faster than np.zeros?
    
    for i in range(0, tParam['nd']):
        for j in range(0, tParam['nIb']):
            if tParam['dConstantUpdate'][i, j] == 1:
                P[i, j] = 1
                break
    
    return Q, P


