import numpy as np
import pandas as pd
import sys
from io import StringIO

import subprocess
import multiprocessing
import time

def mean(P):
    P = pd.read_csv(P)
    P = np.array(P).T
    return P.mean(axis=1)

def dotproduct(P, Q):
    P = pd.read_csv(P)
    Q = pd.read_csv(Q)
    P = np.array(P)
    Q = np.array(Q).T
    return P.dot(Q)

def exec(fn, *args):
    print(*args)
    expected = globals()[fn](*args)
    executable = "./testlibalg{}".format(fn)
    myprocess = subprocess.Popen([executable, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = myprocess.communicate()[0].decode('utf-8')
    R = np.array(pd.read_csv(StringIO(stdout), sep=",", header=None))
    #input(R)
    #input(expected)
    assert(np.allclose(expected, R))
    print("Success !")

if __name__ == "__main__":
    fn = sys.argv[-1]
    if not fn in globals():
        exit(1)
    execargs = [sys.argv[1]]
    if (len(sys.argv) == 4):
        execargs.append(sys.argv[2])
    print(execargs)
    exec(fn, *execargs)