"""
Translation of MATLAB code 'compass_Qk.m' to python
17 Nov 2022  Translated by Jazlin Taylor
1st Dec 2022 Tested & Modified by Sumedh Nagrale
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
1. Removed the loops which were replaced by np.where
2. Optimized the code for speed.
3. Added variables separately

11th Jan 2024
Following modifications were made to
1) Testing the code for validation
"""
import numpy as np


def compass_Qk(tIn=None, tParam=None):
    """
    tParam.nd = number of continuous
    tParam.nx = number of hidden state
    tParam.dLinkMap, the link function between X and continuous variable
    """
    dLinkMap = tParam['dLinkMap'].astype(int)
    dConstantUpdate = tParam['dConstantUpdate']
    q_matrix = np.array([np.where(dLinkMap, tIn[k, dLinkMap], 1) for k in range(len(tIn))])
    p_matrix = np.where(dConstantUpdate, 1.0, 0.0)
    return q_matrix, p_matrix


'''
Param = {'nd': 3, 'nIb': 3, 'xM': np.array([1, 6, 8]), 'dLinkMap': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]]),
         'dConstantUpdate': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]])}
[P, Q] = compass_Qk(tIn=np.array([[10253, 65, 3, 9, 67, 98], [1234, 5, 3, 9, 67, 98]]), tParam=Param)
print(P)
print(Q)
'''
