#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 

'''
The SAX representation, proposed in

Jessica Lin, Eamonn J. Keogh, Li Wei, Stefano Lonardi:
Experiencing SAX: a novel symbolic representation of time series. Data Min. Knowl. Discov. 15(2): 107-144 (2007)

This implementation utilizes the efficient SAX transformation method proposed in

Xiaosheng Li, Jessica Lin:
Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features. ICDM 2017: 277-286

which is based on

Abdullah Mueen, Eamonn J. Keogh, Neal E. Young:
Logical-shapelets: an expressive primitive for time series classification. KDD 2011: 1154-1162

'''

from Discretization.Discretizer import Discretizer
import numpy as np
from Util import GeneralUtil as gu
from copy import deepcopy

class SAX(Discretizer):
    
    def __init__(self, winLen, wordSize, card, meanNorm = True, stdNorm = True, binSizeTh = 3, step = -1):
        super().__init__(winLen, wordSize, card, binSizeTh, step)
        self.type = 'SAX'
        self.meanNorm = meanNorm
        self.stdNorm = stdNorm
        if self.meanNorm and self.stdNorm:
            self.avg = 0
            self.stdv = 1
        self.segStarts = gu.getSegStarts(self.winLen, self.wordSize)
        self.segSizes = self.segStarts[1 :] - self.segStarts[: len(self.segStarts) - 1]
    
    def getCumSums_Ts(self, ts):
        return gu.getCumSums_1D(ts)
    
    def transformTs(self):
        pass    #incremental transformation
    
    def transformSub(self, cumSums, cumSums_2, pos):
        
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
        
        startPts = self.segStarts[: len(self.segStarts) - 1] + pos
        finishPts = self.segStarts[1 :] + pos
        sumSegs = cumSums[finishPts] - cumSums[startPts]
        transformedSub = (sumSegs / self.segSizes - meanSub) / sigmaSub
        return transformedSub
    
    def transformTsFromCumSums(self, cumSums, cumSums_2, poses = None, keepVacancy = False):

        if poses is None or keepVacancy:
            numSub = len(cumSums) - self.winLen
        else:
            numSub = len(poses)
        if poses is None:
            poses = range(numSub)

        transformedTs = np.zeros((numSub, self.wordSize))
        if keepVacancy:
            for pos in poses:
                transformedTs[pos][:] = self.transformSub(cumSums, cumSums_2, pos)
        else:
            for i, pos in enumerate(poses):
                transformedTs[i][:] = self.transformSub(cumSums, cumSums_2, pos)
        #transformedTs = np.around(transformedTs, 2) 
        return transformedTs 
    
    def transfromTssFromCumSums(self, allCumSums, allCumSums_2, tsLens = None, stride = 1, keepVacancy = False, returnPoses = False):
 
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
            transformedTss.append(self.transformTsFromCumSums(allCumSums[i], allCumSums_2[i], poses, keepVacancy))
            if returnPoses:
                allPoses.append(poses)
        if returnPoses:
            return transformedTss, allPoses
        return transformedTss
    
    def discretizeTssFromCumSums_LNR(self, allCumSums, allCumSums_2, tsLens, labels = None, boundStrategy = 'GD', orderStrategy = 'Default'):
        "discretization with locality-aware numerosity reduction (binary-search-like method)"
        
        transformedTss, allPoses = self.transfromTssFromCumSums(allCumSums, allCumSums_2, tsLens, self.winLen, True, True)
        discretizedTss = self.discretizeTransformedDataset_(transformedTss, tsLens, labels, boundStrategy, orderStrategy, False, allPoses, True)
        for i, discretizedTs in enumerate(discretizedTss):
            poses = allPoses[i]
            cumSums = allCumSums[i]
            cumSums_2 = allCumSums_2[i]
            for j in range(len(poses) - 1):
                start = poses[j]
                finish = poses[j + 1] + 1
                localDiscretized = discretizedTs[start : finish]
                discretizedTs[start : finish] = self.fillLocalVacancies(localDiscretized, start, cumSums, cumSums_2)
            discretizedTss[i] = discretizedTs
        return discretizedTss
    
    def discretizeTsFromCumSums_LNR(self, cumSums, cumSums_2, tsLen):

        finish = tsLen - self.winLen
        poses = list(range(0, finish + 1, self.winLen))
        if poses[-1] != finish:
            poses.append(finish)
        transformedTs = self.transformTsFromCumSums(cumSums, cumSums_2, poses, True)
        discretizedTs = self.discretizeTransformedTs(transformedTs, poses, True)
        for i in range(len(poses) - 1):
            start = poses[i]
            finish = poses[i + 1] + 1
            localDiscretized = discretizedTs[start : finish]
            discretizedTs[start : finish] = self.fillLocalVacancies(localDiscretized, start, cumSums, cumSums_2)
        return discretizedTs          
        
    def fillLocalVacancies(self, localDiscretized, offset, cumSums, cumSums_2):
        'binary-search-like method'
        
        ret = deepcopy(localDiscretized)
        numRet = len(ret)
        if numRet < 3:
            return ret
        
        if localDiscretized[0] == localDiscretized[-1]:
            ret[1 : numRet - 1] = ret[0]
            return ret
        
        mid = numRet // 2
        transformed = self.transformSub(cumSums, cumSums_2, mid + offset)  
        ret[mid] = self.discretizeTransformed(transformed)
        ret_l = self.fillLocalVacancies(ret[: mid + 1], offset, cumSums, cumSums_2)
        ret_r = self.fillLocalVacancies(ret[mid:], offset + mid, cumSums, cumSums_2)
        ret[: mid] = ret_l[: mid]
        ret[mid :] = ret_r
        return ret
