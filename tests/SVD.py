import numpy as np
import pandas as pd
import sys

def svd(cov):
    U, S, V_T = np.linalg.svd(cov, full_matrices=False)
    #return V_T.T, S, U.T
    return U, S, V_T

file = sys.argv[1]
file = pd.read_csv(file)
file = np.array(file)

print("A:", file)
U, S, V_T = svd(file)

print("shape: ", U.shape, S.shape, V_T.shape)

print("U:", U)
print("S:", S)
print("VT:", V_T)

# this is inverted compared to numpy...

R = np.dot(U * S, V_T)
print("R: ", R)