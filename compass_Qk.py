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
"""
import numpy as np


def compass_Qk(tIn=None, tParam=None):
    """
    tParam.nd = number of continuous
    tParam.nx = number of hidden state
    tParam.dLinkMap, the link function between X and continuous variable
    """
    q_matrix = []
    for k in range(0, len(tIn)):
        temp = np.ones((tParam['nd'], len(tParam['xM'])))
        for i in range(0, tParam['nd']):
            for j in range(0, len(tParam['xM'])):
                if tParam['dLinkMap'][i, j]:
                    temp[i, j] = tIn[k, tParam['dLinkMap'][i, j]]
        q_matrix.append(temp)

    p_matrix = np.zeros((tParam['nd'], tParam['nIb']))
    for i in range(0, tParam['nd']):
        for j in range(0, tParam['nIb']):
            if tParam['dConstantUpdate'][i, j] == 1:
                p_matrix[i, j] = 1
    return q_matrix, p_matrix

"""
Param = {'nd': 3, 'nIb': 3, 'xM': np.array([1, 6, 8]), 'dLinkMap': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]]),
         'dConstantUpdate': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]])}
[P, Q] = compass_Qk(tIn=np.array([[10253, 65, 3, 9, 67, 98], [1234, 5, 3, 9, 67, 98]]), tParam=Param)
print(P)
print(Q)
"""
