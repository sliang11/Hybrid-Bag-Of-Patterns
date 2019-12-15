#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 

import numpy as np
import sklearn.feature_selection as fs
import Util.GeneralUtil as gu
    
def entropy(c, total):
    p = c / total
    p = p[np.nonzero(p)]
    return -np.sum(p * np.log2(p))
    
def infoGain(cIn, cOut, entAll, total, nIn, nOut):
    ent = entAll
    if nIn > 0:
        ent -= nIn / total * entropy(cIn, nIn)
    if nOut > 0:
        ent -= nOut / total *  entropy(cOut, nOut)    
    return ent

def infoGain_singleSplit(vals, labels, retMajorClasses = False):
        
        #takes in np.array
        
        if len(np.unique(vals)) == 1:   #no distinguishing power at all
            if retMajorClasses:
                return (-1, -1, -1)
            return (-1, -1)
        
        total = len(vals)
        order = np.argsort(vals)
        sortedVals = vals[order]
        sortedLabels = labels[order]
        
        bestGain = -1
        bestPos = -1
        
        uniqLabels, cOut = np.unique(sortedLabels, return_counts = True)
        numCls = len(np.unique(sortedLabels))
        
        labelMap = {}
        for i in range(numCls):
            labelMap[uniqLabels[i]] = i
        
        entAll = entropy(cOut, total)
        
        lastCVal = sortedVals[0]
        nOut = total
        nIn = 0
        cIn = np.zeros(numCls)
        
        for split in range(total):
            cVal = sortedVals[split]
            
            if lastCVal != cVal:
                gain = infoGain(cIn, cOut, entAll, total, nIn, nOut)
                if gain >= bestGain:
                    bestPos = split
                    bestGain = gain
                lastCVal = cVal
                
            labelIdx = labelMap[sortedLabels[split]]
            cOut[labelIdx] -= 1
            nOut -= 1
            cIn[labelIdx] += 1
            nIn += 1
        
        splitPt = sortedVals[bestPos]
        if retMajorClasses:
            labelsOut = sortedLabels[sortedVals >= splitPt]
            uniqLabelsOut, cOut = np.unique(labelsOut, return_counts = True)
            maxC, maxCIdx = gu.maxWithTies(cOut)
            majorClasses = uniqLabelsOut[maxCIdx]
            return bestGain, splitPt, majorClasses
        return bestGain, splitPt

def FStat(vals, labels):
    vals = np.array(vals)
    if len(np.shape(vals)) == 1:
        (f, p) = fs.f_classif(vals.reshape(-1, 1), labels)
        return f[0]
    else:
        (f, p) = fs.f_classif(vals, labels)
        return f
    
def FStat_2(vals, labels, numCls):
    'Must be relabeled to 0 ~ numCls - 1'
    
    vals = np.array(vals)
    num = len(vals)
    s = np.sum(vals)
    s2 = np.sum(vals ** 2)
    minuend = s * s / num
    ss_t = s2 - minuend
    ss_g = - minuend
    for label in range(numCls):
        inds = np.where(labels == label)[0]
        curNum = len(inds)
        ss_g += np.sum(vals[inds]) ** 2 / curNum
    ss_w = ss_t - ss_g
    if ss_w == 0:
        return 0
    
    f = ss_g * (num - numCls) / (ss_w * (numCls - 1))
    return f
    
    
def chi2(vals, labels):
    vals = np.array(vals)
    if len(np.shape(vals)) == 1:
        (chi, p) = fs.chi2(vals.reshape(-1, 1), labels)
        return chi[0]
    else:
        (chi, p) = fs.chi2(vals, labels)
        return chi

