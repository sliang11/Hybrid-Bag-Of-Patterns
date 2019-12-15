#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 

def getSubIndsInDataset(tsLens, winLen, step, repeats = 1):   #supports non-uniform time series lengths
    inds = []
    cumNumInds = 0
    for tsLen in tsLens:
        numSub = tsLen - winLen + 1
        curInds = []
        for pos in range(0, numSub, step):
            curInds += [cumNumInds + pos * repeats + i for i in range(repeats)]
        cumNumInds += numSub * repeats
        inds.append(curInds)
    return inds