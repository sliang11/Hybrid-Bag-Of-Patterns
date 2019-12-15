#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 

import numpy as np

def getCumSums(x):  #Fast version. Pad zero to first 
    x = np.array(x)
    if x.ndim == 1:
        ret = np.concatenate((np.array([0.]), np.cumsum(x)))
    else:
        ret = np.concatenate((np.zeros((x.shape[0], 1)), np.cumsum(x, axis = 1)), axis = 1)
    return ret
        
    
def getCumSums_1D(x):   #Too slow!   
    s = 0
    ret = [0]
    for i in range(len(x)):
        s += x[i]
        ret.append(s)
    return np.array(ret)

def getCumSums_MultiDim(X): #Too slow!
    return np.array([getCumSums_1D(x) for x in X])

def getCumSums_slow(X):
    if len(np.shape(X)) == 1:
        return getCumSums_1D(X)
    else:
        return getCumSums_MultiDim(X)

def getSegStarts(vecLen, numSeg):
    winLen = vecLen // numSeg
    starts = np.arange(0, winLen * numSeg + 1, winLen, dtype = 'int32')
    if starts[-1] != vecLen:
        starts[-1] = vecLen
    return starts

def getAriSeqSum(starts, finishes, steps = 1):
    num = (finishes - starts) // steps + 1
    finishes = starts + (num - 1) * steps   #auto correct
    return (starts + finishes) * num / 2

def getSumOfSquares_first_n(ns):
    ret = ns * (ns + 1) * (2 * ns + 1) / 6
    ret[np.where(ns <= 0)] = 0
    return ret

def getSumOfSquares(starts, finishes):
    return getSumOfSquares_first_n(finishes) - getSumOfSquares_first_n(starts - 1)

def maxWithTies(x):
    x = np.array(x)
    maxX = np.amax(x)
    return maxX, np.where(x == maxX)

def minWithTies(x):
    x = np.array(x)
    minX = np.amin(x)
    return minX, (np.where(x = minX))[0]