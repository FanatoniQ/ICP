import numpy as np
import pandas as pd
import sys
from io import StringIO

import subprocess
import multiprocessing
import time
import glob

success = 0

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
    print("  ===================\n" + ENDC)

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
    global success
    print(OKBLUE + fn + str(args) + ENDC)
    ret = True
    try:
        expected = globals()[fn](*args)
    except Exception:
        ret = False
    executable = "./testlibalg{}".format(fn)
    myprocess = subprocess.Popen([executable, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    myprocess.communicate()

    pret = bool(myprocess.returncode == 0)
    if (pret != ret):
        print(myprocess.returncode)
        print(ret)
        print(FAIL + "Fail return code !\n" + ENDC)
        return
    
    if ret:
        stdout = myprocess.communicate()[0].decode('utf-8')
        R = np.array(pd.read_csv(StringIO(stdout), sep=",", header=None))
        try:
            if (np.allclose(expected, R)):
                print(OKGREEN + "Success !\n" + ENDC)
                success += 1
            else:
                print(FAIL + "Fail (not equal) !\n" + ENDC)
        except Exception as e:
            print(FAIL + "Fail (different shape) !")
            print(e)
            print(ENDC)
    else:
        print(OKGREEN + "Success (invalid operation in both cases) !\n" + ENDC)
        success += 1

if __name__ == "__main__":
    print_test('TESTSUITE')
    nb_tests = 0

    fileList = glob.glob('../data/*.txt')
    params_1fn = [
        "mean"
    ]
    params_2fn = [
        "dotproduct"
    ]
    for fn in params_1fn:
        for file in fileList:
            exec(fn, file)
            nb_tests += 1
    for fn in params_2fn:
        for i in range(len(fileList)):
            for j in range(i, len(fileList)):
                exec(fn, fileList[i],fileList[j])
                nb_tests += 1

    end_print(nb_tests, success, nb_tests - success)