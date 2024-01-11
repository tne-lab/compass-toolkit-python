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


def compass_Tk(tIn, tParam):
    """
    Build the T matrix based on the input data and parameters.

    Args:
    tIn (numpy.ndarray): Input data.
    tParam (dict): Parameters including 'nc', 'xM', 'cLinkMap', 'cConstantUpdate'.
    tParam.nd = number of continuous
    tParam.nx = number of hidden state
    tParam.dLinkMap, the link function between X and continuous variable

    Returns:
    list of numpy.ndarray: List of T matrices.
    numpy.ndarray: P matrix.
    """
    # Variables
    cLinkMap = tParam['cLinkMap'].astype(int)
    cConstantUpdate = tParam['cConstantUpdate']

    # Create a list of T matrices using a list comprehension
    t_matrix = np.array([np.where(cLinkMap, tIn[k, cLinkMap], 1) for k in range(len(tIn))])

    # Create the P matrix using NumPy operations
    p_matrix = np.where(cConstantUpdate, 1.0, 0.0)
    return t_matrix, p_matrix


'''
# Test case: Single state
Param = {'nc': 3, 'nIn': 3, 'xM': np.array([1, 6, 8]), 'cLinkMap': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]]),
         'cConstantUpdate': np.array([[0, 1, 0], [1, 1, 0], [1, 1, 1]])}
[T, P] = compass_Tk(tIn=np.array([[10253, 65, 3, 9, 67, 98], [1234, 5, 3, 9, 67, 98]]), tParam=Param)
print(T)
print(P)
'''
