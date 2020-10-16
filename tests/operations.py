import numpy as np
import pandas as pd

from io import StringIO


"""
simple decorator to convert a two file operator to np array
"""
def twofileconvert(func):
    def function_wrapper(P, Q):
        return func(np.array(pd.read_csv(P)),np.array(pd.read_csv(Q)))
    return function_wrapper

def onefileconvert(func):
    def function_wrapper(P, *args):
        return func(np.array(pd.read_csv(P)), *args)
    return function_wrapper


"""
svd function not used
def svd(cov):
    cov = pd.read_csv(cov)
    U, S, V_T = np.linalg.svd(cov, full_matrices=False)
    return V_T.T, S, U.T # dumb
"""

"""
svd recomposition used for svd checking
"""
def svd_no_ref(u, s, vh):
    u, s, vh = StringIO(u), StringIO(s), StringIO(vh)
    u, s, vh = pd.read_csv(u, sep=",", header=None), pd.read_csv(s, sep=",", header=None), pd.read_csv(vh, sep=",", header=None)
    u, s, vh = np.array(u), np.array(s), np.array(vh)
    u, s, vh = np.squeeze(u),np.squeeze(s),np.squeeze(vh)
    print(u.shape, s.shape, vh.shape)
    return np.dot(u * s, vh)

"""
sum axis computation
"""
@onefileconvert
def sum(P, axis):
    axis = int(axis)
    if axis == -1:
        axis = None
    return P.sum(axis=axis)

"""
mean axis computation
"""
@onefileconvert
def mean(P, axis):
    axis = int(axis)
    if axis == -1:
        axis = None
    return P.mean(axis=axis)

@twofileconvert
def dotproduct(P, Q):
    return P.dot(Q.T)

@twofileconvert
def mult(P, Q):
    return P * Q

@twofileconvert
def divide(P, Q):
    return P / Q

@twofileconvert
def add(P, Q):
    return P + Q

@twofileconvert
def subtract(P, Q):
    return P - Q