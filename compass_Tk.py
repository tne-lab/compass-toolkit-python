'''
Translation of MATLAB code 'compass_Tk.m' to python language
17 Nov 2022 - Jazlin Taylor
'''
import numpy as np

def compass_Tk(tIn, tParam):
    '''
    tParam.nd = number of continous
    tParam.nx = number of hidden state
    tParam.dLinkMap, the link function between X and continous variable
    '''

    T = [None] * len(tIn) # This is T, an array of arrays

    for k in range(0, len(tIn)):
        temp = np.empty((tParam['nc'], len(tParam['xM'])))
        for i in range(0, tParam['nc']):
            for j in range(0, len(tParam['xM'])):
                if tParam['cLinkMap'][i,j]:
                    temp[i, j] = tIn(k, tParam['cLinkMap'][i,j])
                    break
        T[k] = temp
    
    P = np.empty((tParam['nc'], tParam['nIn'])) #np.empty faster than np.zeros?
    
    for i in range(0, tParam['nc']):
        for j in range(0, tParam['nIn']):
            if tParam['cConstantUpdate'][i, j]:
                P[i, j] = 1
                break
    
    return T, P
    



