import numpy as np
import pandas as pd
import sys
from io import StringIO

import subprocess
import multiprocessing
import time

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def print_test(TEST):
    """Pretty print for testsuite start"""
    print(WARNING + "\n  ===================")
    print(" /                   \\")
    print("       " + TEST )
    print(" \\                   /")
    print("  ===================" + ENDC)

def end_print(total, success, fails):
    """Pretty print for testsuite results"""
    print("\nTotal:", total)
    print(OKGREEN + "Success:" + ENDC, success)
    print(FAIL + "Fails:" + ENDC , fails)

#-------------------------------------------------

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
    print_test('TESTSUITE')

    nbtests = 0
    success = 0

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
    success += 1; nbtests += 1
    end_print(1, success, nbtests - success)

if __name__ == "__main__":
    fn = sys.argv[-1]
    if not fn in globals():
        exit(1)
    execargs = [sys.argv[1]]
    if (len(sys.argv) == 4):
        execargs.append(sys.argv[2])
    print(execargs)
    exec(fn, *execargs)