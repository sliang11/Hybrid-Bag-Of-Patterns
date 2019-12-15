#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 
'''
The SLA representation

The efficient PLA transformation is partially based on

Xiaosheng Li, Jessica Lin:
Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features. ICDM 2017: 277-286

and

Abdullah Mueen, Eamonn J. Keogh, Neal E. Young:
Logical-shapelets: an expressive primitive for time series classification. KDD 2011: 1154-1162
'''
from Discretization.Discretizer import Discretizer
import numpy as np
from Util import GeneralUtil as gu
from copy import deepcopy

class SLA(Discretizer):
    
    def __init__(self, winLen, wordSize, card, meanNorm = True, stdNorm = True, posNorm = True, binSizeTh = 3, step = -1): 
        super().__init__(winLen, wordSize, card, binSizeTh, step)
        self.type = 'SLA'
        if self.wordSize % 2:
            self.wordSize += 1  #make it even so that both the slopes and the intercepts can be kept
        self.meanNorm = meanNorm
        self.stdNorm = stdNorm
        self.posNorm = posNorm
        self.segStarts = gu.getSegStarts(self.winLen, self.wordSize / 2)
        self.segSizes = self.segStarts[1 :] - self.segStarts[: len(self.segStarts) - 1]
    
    def transformSub(self, cumSums, cumSums_2, wCumSums, pos):
        
        transformedSub = np.zeros(self.wordSize)
        
        #window mean and std
        if not (self.meanNorm or self.stdNorm):
            meanSub = 0
            sigmaSub = 1
        elif self.stdNorm:
            meanSub = (cumSums[pos + self.winLen] - cumSums[pos]) / self.winLen
            meanSub_2 = (cumSums_2[pos + self.winLen] - cumSums_2[pos]) / self.winLen
            varSub = meanSub_2 - meanSub * meanSub
            sigmaSub = np.sqrt(varSub) if varSub > 0 else 1
            if not self.meanNorm:
                meanSub = 0
        else:
            meanSub = (cumSums[pos + self.winLen] - cumSums[pos]) / self.winLen
            sigmaSub = 1
        
        #timestamp parameters
        startPts = self.segStarts[: len(self.segStarts) - 1] + pos
        finishPts = self.segStarts[1 :] + pos
        sum_X = gu.getAriSeqSum(startPts, finishPts - 1)
        mean_X = sum_X / self.segSizes
        mean_X2 = gu.getSumOfSquares(startPts, finishPts - 1) / self.segSizes
        
        #segment parameters
#             sumSegs = cumSums[self.segStarts[1 :]] - cumSums[self.segStarts[: len(self.segStarts) - 1]]
        sumSegs = cumSums[finishPts] - cumSums[startPts]
        meanSegs = (sumSegs / self.segSizes - meanSub) / sigmaSub
#             wCumSegs = wCumSums[self.segStarts[1 :]] - wCumSums[self.segStarts[: len(self.segStarts) - 1]]
        wCumSegs = wCumSums[finishPts] - wCumSums[startPts]
        wMeanSegs = (wCumSegs - meanSub * sum_X) / self.segSizes / sigmaSub
        
        #the coefficients
        slopes = (wMeanSegs - mean_X * meanSegs) / (mean_X2 - mean_X * mean_X)
        intercepts = meanSegs - slopes * mean_X
        if self.posNorm:
            intercepts += startPts * slopes    #shift to the same starting timestamp of 0
        transformedSub[0 : self.wordSize - 1 : 2] = slopes
        transformedSub[1 : self.wordSize : 2] = intercepts
        return transformedSub
    
    def transformTsFromCumSums(self, cumSums, cumSums_2, wCumSums, poses = None, keepVacancy = False):

#         numSub = len(cumSums) - self.winLen if poses is None else len(poses)
        if poses is None or keepVacancy:
            numSub = len(cumSums) - self.winLen
        else:
            numSub = len(poses)
        if poses is None:
            poses = range(numSub)

        transformedTs = np.zeros((numSub, self.wordSize))
        if keepVacancy:
            for pos in poses:
                transformedTs[pos][:] = self.transformSub(cumSums, cumSums_2, wCumSums, pos)
        else:
            for i, pos in enumerate(poses):
                transformedTs[i][:] = self.transformSub(cumSums, cumSums_2, wCumSums, pos)
        #transformedTs = np.around(transformedTs, 2) 
        return transformedTs
    
    def transfromTssFromCumSums(self, allCumSums, allCumSums_2, allWCumSums, tsLens = None, stride = 1, keepVacancy = False, returnPoses = False):

        transformedTss = []
        if returnPoses:
            allPoses = []
        for i in range(len(allCumSums)):
            if tsLens is None:
                poses = None
            else:
                finish = tsLens[i] - self.winLen
                poses = list(range(0, finish + 1, stride))
                if poses[-1] != finish:
                    poses.append(finish)
            transformedTss.append(self.transformTsFromCumSums(allCumSums[i], allCumSums_2[i], allWCumSums[i], poses, keepVacancy))
            if returnPoses:
                allPoses.append(poses)
        if returnPoses:
            return transformedTss, allPoses
        return transformedTss

    def discretizeTssFromCumSums_LNR(self, allCumSums, allCumSums_2, allWCumSums, tsLens, labels = None, boundStrategy = 'ED', orderStrategy = 'Default'):
        "discretization with locality-aware numerosity reduction (binear-search-like method)"
        transformedTss, allPoses = self.transfromTssFromCumSums(allCumSums, allCumSums_2, allWCumSums, tsLens, self.winLen, True, True)
        discretizedTss = self.discretizeTransformedDataset_(transformedTss, tsLens, labels, boundStrategy, orderStrategy, False, allPoses, True)
        for i, discretizedTs in enumerate(discretizedTss):
            poses = allPoses[i]
            cumSums = allCumSums[i]
            cumSums_2 = allCumSums_2[i]
            wCumSums = allWCumSums[i]
            for j in range(len(poses) - 1):
                start = poses[j]
                finish = poses[j + 1] + 1
                localDiscretized = discretizedTs[start : finish]
                discretizedTs[start : finish] = self.fillLocalVacancies(localDiscretized, start, cumSums, cumSums_2, wCumSums)
            discretizedTss[i] = discretizedTs
        return discretizedTss
    
    def discretizeTsFromCumSums_LNR(self, cumSums, cumSums_2, wCumSums, tsLen):
        
        finish = tsLen - self.winLen
        poses = list(range(0, finish + 1, self.winLen))
        if poses[-1] != finish:
            poses.append(finish)
        transformedTs = self.transformTsFromCumSums(cumSums, cumSums_2, wCumSums, poses, True)
        discretizedTs = self.discretizeTransformedTs(transformedTs, poses, True)
        for i in range(len(poses) - 1):
            start = poses[i]
            finish = poses[i + 1] + 1
            localDiscretized = discretizedTs[start : finish]
            discretizedTs[start : finish] = self.fillLocalVacancies(localDiscretized, start, cumSums, cumSums_2, wCumSums)
        return discretizedTs       
        
    def fillLocalVacancies(self, localDiscretized, offset, cumSums, cumSums_2, wCumSums):
        'binary-search-like method'
        
        ret = deepcopy(localDiscretized)
        numRet = len(ret)
        if numRet < 3:
            return ret
        
        if localDiscretized[0] == localDiscretized[-1]:
            ret[1 : numRet - 1] = ret[0]
            return ret
        
        mid = numRet // 2
        transformed = self.transformSub(cumSums, cumSums_2, wCumSums, mid + offset)
        ret[mid] = self.discretizeTransformed(transformed)
        ret_l = self.fillLocalVacancies(ret[: mid + 1], offset, cumSums, cumSums_2, wCumSums)
        ret_r = self.fillLocalVacancies(ret[mid:], offset + mid, cumSums, cumSums_2, wCumSums)
        ret[: mid] = ret_l[: mid]
        ret[mid :] = ret_r
        return ret
        
            
        
        
        
        
        
    
