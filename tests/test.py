import numpy as np
import pandas as pd
import sys
from io import StringIO

import subprocess
import multiprocessing
import time
import glob

from operations import *

success = 0

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
YELLOW = '\033[33m'
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

"""
exec_no_ref executes a test, for which there is no ref from inputs but
only from the output (svd correctness)
TODO: we could generalize by giving postprocessing functions
"""
def exec_no_ref(fn, *args):
    global success
    print(OKBLUE + fn + str([args]) + ENDC)
    R = np.array(pd.read_csv(args[0], sep=","))
    executable = "./testlibalg" #if fn != "svd" else "./SVD"
    myprocess = subprocess.Popen([executable, fn, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = myprocess.communicate()[0]

    array_delim = "\n\n"
    stdout = stdout.decode('utf-8').rstrip(array_delim).split(array_delim)
    #print([ np.array(pd.read_csv(StringIO(e), sep=",", header=None)).shape for e in stdout ])
    #[::-1]
    expected = globals()["{}_no_ref".format(fn)](*stdout)
    try:
        if not np.allclose(R, expected):
            print(YELLOW, expected)
            print(R, ENDC)
            print(FAIL,"Fail (not equal) !",ENDC)
            print()
        else:
            print(OKGREEN,"Success !",ENDC)
            print()
            success += 1
    except Exception as e:
        print(YELLOW, expected)
        print(R, ENDC)
        print()
        print(FAIL + "Fail (different shape) !")
        print(e)
        print(ENDC)

"""
standard numpy ref checking
"""
def exec(fn, *args):
    if fn == "svd": # this is for now the only no ref needed test
        exec_no_ref(fn, *args)
        return
    global success
    print(OKBLUE + fn + str(args) + ENDC)
    ret = True
    try:
        expected = globals()[fn](*args)
    except Exception as e:
        print(YELLOW, e, ENDC)
        ret = False
    executable = "./testlibalg" # if fn != "svd" else "./SVD"
    myprocess = subprocess.Popen([executable, fn, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = myprocess.communicate()[0]

    pret = bool(myprocess.returncode == 0)
    if (pret != ret):
        print(FAIL,"Fail (return code) !",ENDC)
        print(YELLOW, "we have {} instead of {}".format(myprocess.returncode, 0 if ret else 1), ENDC)
        print()
        return
    
    if ret:
        array_delim = "\n\n"
        stdout = stdout.decode('utf-8').rstrip(array_delim).split(array_delim)
        if not isinstance(expected, tuple):
            expected = [ expected ]
        if len(expected) != len(stdout):
            print(FAIL,"Fail (not same number of returned arrays) " + str(len(stdout)) + "!",ENDC)
            print()
        else:
            s = True
            for i in range(len(expected)):
                R = np.array(pd.read_csv(StringIO(stdout[i]), sep=",", header=None))
                try:
                    if (np.allclose(expected[i], R)):
                        print(OKGREEN,"Success (" + str(i) + ") !",ENDC)
                        print()
                    else:
                        print(FAIL,"Fail (not equal " + str(i) + ") !",ENDC)
                        print(YELLOW, expected[i])
                        print(R, ENDC)
                        print()
                        s = False
                except Exception as e:
                    print(FAIL, "Fail (different shape) !")
                    print(e)
                    print(ENDC)
                    print(YELLOW, expected[i])
                    print(R, ENDC)
                    print()
                    s = False
            success += 1 if s else 0
    else:
        print(OKGREEN, "Success (invalid operation in both cases) !",ENDC)
        print()
        success += 1



if __name__ == "__main__":
    print_test('TESTSUITE')
    nb_tests = 0

    fileList = glob.glob('../data/*.txt')
    params_1fn = [
        ["mean", "-1"],
        ["mean", "0"],
        ["mean", "1"],
        ["sum", "-1"],
        ["sum", "0"],
        ["sum", "1"],
        ["svd"],
    ] # len(fileList) * len(params_1fn)
    params_2fn = [
        ["dotproduct"],
    ] # (len(fileList) * (len(fileList) + 1) / 2) * len(params_2fn)
    for fn in params_1fn:
        for file in fileList:
            exec(fn[0], file, *fn[1:])
            nb_tests += 1
    for fn in params_2fn:
        for i in range(len(fileList)):
            for j in range(i, len(fileList)):
                exec(fn[0], fileList[i], fileList[j], *fn[1:])
                nb_tests += 1

    end_print(nb_tests, success, nb_tests - success)
    exit(nb_tests - success)