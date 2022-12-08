"""
Translation of MATLAB code 'compass_Tk.m' to python
Same as Qk functionally, can therefore be merged together, however for time being
we keep it combined to avoid any confusion
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


def compass_Tk(tIn, tParam):
    """
    tParam.nd = number of continuous
    tParam.nx = number of hidden state
    tParam.dLinkMap, the link function between X and continuous variable
    """
    t_matrix = []  # This is T, an array of arrays
    for k in range(0, len(tIn)):
        temp = np.ones((tParam['nc'], len(tParam['xM'])))
        for i in range(0, tParam['nc']):
            for j in range(0, len(tParam['xM'])):
                if tParam['cLinkMap'][i, j]:
                    temp[i, j] = tIn[k, tParam['cLinkMap'][i, j]]
        t_matrix.append(temp)

    p_matrix = np.zeros((tParam['nc'], tParam['nIn']))
    for i in range(0, tParam['nc']):
        for j in range(0, tParam['nIn']):
            if tParam['cConstantUpdate'][i, j]:
                p_matrix[i, j] = 1

    return t_matrix, p_matrix

"""
# Test case: Single state
Param = {'nc': 3, 'nIn': 3, 'xM': np.array([1, 6, 8]), 'cLinkMap': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]]),
         'cConstantUpdate': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]])}
[T, P] = compass_Tk(tIn=np.array([[10253, 65, 3, 9, 67, 98], [1234, 5, 3, 9, 67, 98]]), tParam=Param)
print(T)
print(P)
"""
