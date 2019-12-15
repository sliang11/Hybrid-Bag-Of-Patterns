#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 
'''
Time series discretization
'''
import numpy as np
from TimeSeries import TimeSeries
from Util import EvaluationUtil as evalUtil
from Util import BitUtil as bu
from Util import GeneralUtil as gu
from copy import deepcopy

class Discretizer(object):
    def __init__(self, winLen, wordSize, card, binSizeTh = 3, step = -1):
        
        self.winLen = winLen
        self.wordSize = wordSize
        self.oriWordSize = wordSize
        self.card = card
        self.binSizeTh = binSizeTh
        self.step = step if step > 0 else winLen
        self.allBoundaries = None
        self.cValOrder = None
        self.groups = None
        
    def reset(self, winLen = None, wordSize = None, card = None):
        self.allBoundaries = None
        self.cValOrder = None
        if winLen is not None:
            self.winLen = winLen
        if wordSize is not None:
            self.wordSize = wordSize
        if card is not None:
            self.card = card
        
    def transformTs(self, ts):    #to override
        return None
    
    def transformDataset(self, tss):
        transformedTss = [self.transformTs(ts) for ts in tss]
#         transformedTss = np.around(transformedTss, 2)
        return np.array(transformedTss)
    
    def setAllBoundaries(self, allCVals, indices, strategy, labels = None):
        
        cols = np.shape(allCVals)[1]
        if self.groups is None:
            numLetters = cols
        else:
            numGroups = len(self.groups)
            numLetters = np.sum([len(group) for group in self.groups])
        if strategy == 'GD':    #Gaussian distribution
            if self.groups is None:
                boundaries = self.setBoundaries_Guassian(self.avg, self.stdv)
                uniqBoundaries = [boundaries] * numLetters
            else:
                uniqBoundaries = []
                for i in range(numGroups):
                    uniqBoundaries.append(self.setBoundaries_Guassian(self.avgs[i], self.stdvs[i]))
        else:
            if strategy == 'IG':
                curLabels = labels[indices]
            uniqBoundaries = []
            for i in range(cols):
                cVals = allCVals[indices, i]
                if len(np.unique(cVals)) == 1:
                    boundaries = [float('-inf'), float('inf')]
                elif strategy == 'ED': #equal depth
                    boundaries = self.setBoundaries_EqualDepth_Fast(cVals)
                elif strategy == 'IG': #info gain
                    boundaries = self.setBoundaries_InfoGain(cVals, curLabels)
                uniqBoundaries.append(boundaries)
        if self.groups is None:
            self.allBoundaries = uniqBoundaries
        else:
            self.allBoundaries = [[] for i in range(numLetters)]
            for i in range(numGroups):
                for ind in self.groups[i]:
                    self.allBoundaries[ind] = uniqBoundaries[i]            
    
    def setBoundaries_Guassian(self, avg = 0, stdv = 1):
        '''
        For SAX:
        Jessica Lin, Eamonn J. Keogh, Li Wei, Stefano Lonardi:
        Experiencing SAX: a novel symbolic representation of time series. Data Min. Knowl. Discov. 15(2): 107-144 (2007)
        '''
        
        boundaries = None
        if self.card == 3:
            boundaries = [-0.43, 0.43]
        if self.card == 4:
            boundaries = [-0.67, 0, 0.67]
        if self.card == 5:
            boundaries = [-0.84, -0.25, 0.25, 0.84]
        if self.card == 6:
            boundaries = [-0.97, -0.43, 0, 0.43, 0.97]
        if self.card == 7:
            boundaries = [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07]
        if self.card == 8:
            boundaries = [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15]
        if self.card == 9:
            boundaries = [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22]
        if self.card == 10:
            boundaries = [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28]
        if stdv <= 0:
            stdv = 1
        boundaries = np.around(np.array(boundaries) * stdv + avg, 2).tolist()
        boundaries = [float('-inf')] + boundaries + [float('inf')]
        return boundaries
    
    def setBoundaries_EqualDepth(self, cVals):
        
        '''
        Equi-depth-bining (slow)
        Patrick Schafer, Mikael Hogqvist:
        SFA: a symbolic fourier approximation and index for similarity search in high dimensional datasets. EDBT 2012: 516-527
        '''
        
        numCVal = len(cVals)
        binSize = numCVal // self.card
        if binSize < self.binSizeTh:
            binSize = self.binSizeTh
        boundaryInds = np.arange(binSize, numCVal - binSize + 1, binSize)
        sortedCVals = np.sort(cVals)
        return self.applyBoundaries(sortedCVals, boundaryInds)
    
    def setBoundaries_EqualDepth_Fast(self, cVals):
        '''
        Equi-depth-bining (fast, with selection algorithm)
        
        Patrick Schafer, Mikael Hogqvist:
        SFA: a symbolic fourier approximation and index for similarity search in high dimensional datasets. EDBT 2012: 516-527
        '''
        
        numCVal = len(cVals)
        binSize = numCVal // self.card
        if binSize < self.binSizeTh:
            binSize = self.binSizeTh
        boundaryInds = np.arange(binSize, numCVal - binSize + 1, binSize)
        if boundaryInds[-1] >= numCVal - self.binSizeTh + 1:
            boundaryInds = boundaryInds[: len(boundaryInds) - 1]
        partitioned = np.partition(cVals, tuple(boundaryInds))
        boundaries = np.concatenate((np.array([float('-inf')]), partitioned[boundaryInds], np.array([float('inf')])))
        return np.unique(boundaries)
    
    def setBoundaries_InfoGain(self, cVals, labels):
        '''
        Supervised discretization:
        
        Patrick Schafer, Ulf Leser:
        Fast and Accurate Time Series Classification with WEASEL. CIKM 2017: 637-646
        
        This is not used in the final version of HBOP for efficiency concerns.
        '''
        boundaryInds = []
        order = np.argsort(cVals)
        sortedCVals = cVals[order]
        sortedLabels = labels[order]
        self.splitOrderline(boundaryInds, sortedCVals, sortedLabels, 0, len(labels), self.card)
        return self.applyBoundaries(sortedCVals, boundaryInds)
    
    def splitOrderline(self, boundaryInds, sortedCVals, sortedLabels, start, end, numRemainingBins):
        
        total = end - start
        if total <= self.binSizeTh:
            return
        
        bestGain = -1
        bestPos = -1
        
        uniqLabels, cOut = np.unique(sortedLabels, return_counts = True)
        numCls = len(np.unique(sortedLabels))
        if numCls == 1: #purity = 1
            return
        
        labelMap = {}
        for i in range(numCls):
            labelMap[uniqLabels[i]] = i
        
        entAll = evalUtil.entropy(cOut, total)
        
        lastCVal = sortedCVals[start]
        nOut = total
        nIn = 0
        cIn = np.zeros(numCls)
        
        for split in range(start, end):
            cVal = sortedCVals[split]
            
            if lastCVal != cVal:
                gain = evalUtil.infoGain(cIn, cOut, entAll, total, nIn, nOut)
                if gain >= bestGain:
                    bestPos = split
                    bestGain = gain
                lastCVal = cVal
                
            labelIdx = labelMap[sortedLabels[split]]
            cOut[labelIdx] -= 1
            nOut -= 1
            cIn[labelIdx] += 1
            nIn += 1
            
        if bestPos > -1:
            boundaryInds.append(bestPos)
        else:
            return
            
        numRemainingBins //= 2
        if numRemainingBins > 1:
            if bestPos - start >= self.binSizeTh and end - bestPos >= self.binSizeTh:
                self.splitOrderline(boundaryInds, sortedCVals, sortedLabels, start, bestPos, numRemainingBins)
                self.splitOrderline(boundaryInds, sortedCVals, sortedLabels, bestPos, end, numRemainingBins)
            elif end - bestPos > 2 * self.binSizeTh:
                self.splitOrderline(boundaryInds, sortedCVals, sortedLabels, bestPos, bestPos + (end - bestPos) // 2, numRemainingBins) 
                self.splitOrderline(boundaryInds, sortedCVals, sortedLabels, bestPos + (end - bestPos) // 2, end, numRemainingBins)
            elif bestPos - start >  2 * self.binSizeTh:
                self.splitOrderline(boundaryInds, sortedCVals, sortedLabels, start, start + (bestPos - start) // 2, numRemainingBins)
                self.splitOrderline(boundaryInds, sortedCVals, sortedLabels, start + (bestPos - start) // 2, bestPos, numRemainingBins)      
    
    def applyBoundaries(self, sortedCVals, boundaryInds):
        boundaries = [float('-inf'), float('inf')]
        for ind in boundaryInds:
            if ind < len(sortedCVals) - self.binSizeTh + 1:
                boundaries.append(sortedCVals[ind])
        boundaries = np.sort(np.unique(boundaries))
        return boundaries
    
    def setCValOrder(self, allCVals = None, labels = None, indices = None, strategy = 'Default'):
        
        if strategy == 'Default':
            self.cValOrder = range(self.wordSize)
        else:
            if strategy == 'ANOVA':
                self.cValOrder = self.setCValOrder_ANOVA(allCVals[indices], labels[indices])
            if self.groups is not None: 
                self.cValOrder = self.groups[self.cValOrder].flatten()
            
    def setCValOrder_ANOVA(self, allCVals, labels):
        '''
        Patrick Schafer, Ulf Leser:
        Fast and Accurate Time Series Classification with WEASEL. CIKM 2017: 637-64
        '''
        
        f = evalUtil.FStat(allCVals, labels)
        return np.argsort(-f)
    
    def discretizeTransformed(self, transformed):
        numBitsLetter = bu.numBits(self.card)
        word = 0
        wordSize = min([self.wordSize, len(self.cValOrder)])
        for i in range(wordSize):
            idx = self.cValOrder[i]
            val = transformed[idx]
            boundaries = self.allBoundaries[idx]
            for j in range(0, len(boundaries) - 1):
                if val >= boundaries[j] and val < boundaries[j + 1]:
                    word = bu.appendBits(word, j, numBitsLetter) 
                    break
        return word
    
    def discretizeTransformedTs(self, transformedTs, poses = None, keepVacancy = False):
        
        if poses is None:
            discretizedTs = [self.discretizeTransformed(transformed) for transformed in transformedTs]
        else:
            poses = np.array(poses)
            if keepVacancy:
                discretizedTs = [0 for i in range(len(transformedTs))]
                for pos in poses:
                    discretizedTs[pos] = self.discretizeTransformed(transformedTs[pos])
            else:
                discretizedTs = [self.discretizeTransformed(transformed) for transformed in transformedTs[poses]]
            
        return np.array(discretizedTs)
    
    def discretizeTransformedDataset(self, transformedTss, allPoses = None, keepVacancy = False):
        
        if allPoses is None:
            discretizedTss = [self.discretizeTransformedTs(transformedTs) for transformedTs in transformedTss]
        else:
            discretizedTss = []
            for i, transformedTs in enumerate(transformedTss):
                discretizedTss.append(self.discretizeTransformedTs(transformedTs, allPoses[i], keepVacancy))
        return np.array(discretizedTss)
    
    def discretizeTs(self, ts): #Pre-set the boundaries and order.
        transformedTs = self.transformTs(ts)
        return self.discretizeTransformedTs(transformedTs)
    
    def regroup(self, allCVals, allLabels): 
        
        (cols, numPerGroup) = np.shape(self.groups)
        rows = len(allCVals) * numPerGroup 
        if allLabels is None:
            regroupedAllLabels = None
        else:
            regroupedAllLabels = np.repeat(allLabels, numPerGroup)
        regroupedAllCVals= np.empty((rows, cols))
        for i in range(cols):
            inds = self.groups[i]
            regroupedAllCVals[:, i] = np.reshape(allCVals[:, inds], rows)
        return regroupedAllCVals, regroupedAllLabels
    
    def discretizeTransformedDataset_(self, transformedTss, tsLens, labels = None, boundStrategy = 'ED', orderStrategy = 'Default', needRegroup = True, allPoses = None, keepVacancy = False):
        
        allCVals = []
        for transformedTs in transformedTss:
            for transformed in transformedTs:
                allCVals.append(transformed)
        allCVals = np.array(allCVals)
        
        if self.allBoundaries is None or self.cValOrder is None:
            numTs = len(tsLens)
            if labels is not None:
                allLabels = []
                allNumSub = [tsLen - self.winLen + 1 for tsLen in tsLens]
                for i in range(numTs):
                    allLabels += [labels[i]] * allNumSub[i]
                allLabels = np.array(allLabels)
            else:
                allLabels = None
            
            needRegroup = needRegroup and self.groups is not None
            if needRegroup:               
                allCVals, allLabels = self.regroup(allCVals, allLabels)
                repeats = np.shape(self.groups)[1]
            else:
                repeats = 1
            indices = TimeSeries.getSubIndsInDataset(tsLens, self.winLen, self.step, repeats)
            flatIndices = []
            for i in range(numTs):
                flatIndices += indices[i]
            if self.allBoundaries is None:  
                self.setAllBoundaries(allCVals, flatIndices, boundStrategy, allLabels)
            if self.cValOrder is None:
                self.setCValOrder(allCVals, allLabels, flatIndices, orderStrategy)
        return self.discretizeTransformedDataset(transformedTss, allPoses, keepVacancy)
        
    def discretizeDataset(self, tss, labels = None, tsLens = None, boundStrategy = 'ED', orderStrategy = 'Default'):
        transformedTss = self.transformDataset(tss)
        if tsLens is None:
            tsLens = [len(ts) for ts in tss]
        return self.discretizeTransformedDataset_(transformedTss, tsLens, labels, boundStrategy, orderStrategy)
    
    def getNewDiscretizer(self, deltaWordSize):
        if not deltaWordSize:
            return deepcopy(self)
        
        newDiscretizer = deepcopy(self)
        newDiscretizer.wordSize = self.wordSize - deltaWordSize
        if self.cValOrder is not None:
            newDiscretizer.cValOrder = deepcopy(self.cValOrder[: newDiscretizer.wordSize])
        return newDiscretizer
    
    def calcDistanceBetweenWords(self, word_0, word_1, sfxLen = 0):
        word_0 = bu.trimBits(word_0, sfxLen)
        word_1 = bu.trimBits(word_1, sfxLen)
        numBitsLetter = bu.numBits(self.card)
        dist = 0
        for i in range(self.wordSize):
            shift  = i * numBitsLetter
            letter_0 = bu.getBits(word_0, shift, numBitsLetter)
            letter_1 = bu.getBits(word_1, shift, numBitsLetter)
            dist += np.abs(letter_0 - letter_1)
        return dist

def getAllCumSums(data):

    cumSums = gu.getCumSums(data)
    cumSums_2 = gu.getCumSums(data * data)
    weightedCumSums = gu.getCumSums(data * np.arange(data.shape[-1]))
    return (cumSums, cumSums_2, weightedCumSums)

def getAllMeanAndStdSub(cumSums, cumSums_2, winLen):
    '''
    Abdullah Mueen, Eamonn J. Keogh, Neal E. Young:
    Logical-shapelets: an expressive primitive for time series classification. KDD 2011: 1154-1162
    '''
    
    shape = np.shape(cumSums)
    if len(shape) == 1:
        numSub = shape[0] - winLen
        meanSub = (cumSums[winLen :] - cumSums[: numSub]) / winLen
        meanSub_2 = (cumSums_2[winLen :] - cumSums_2[: numSub]) / winLen
        varSub = meanSub_2 - meanSub * meanSub
        varSub[np.where(varSub <= 0)] = 1
        stdSub = np.sqrt(varSub)
        return meanSub, stdSub
    else:
        meanSub = []
        stdSub = []
        for i in range(shape[0]):
            curMeanSub, curStdSub = getAllMeanAndStdSub(cumSums[i], cumSums_2[i], winLen)
            meanSub.append(curMeanSub)
            stdSub.append(curStdSub)
        return np.array(meanSub), np.array(stdSub)

def getAllSumX(tsLen, segSize):
    
    first = gu.getAriSeqSum(0, segSize - 1, 1)
    return np.arange(first, first + segSize * (tsLen - segSize) + 1, segSize)

def getAllMeanX2(tsLen, segSize):
    starts = np.arange(tsLen - segSize + 1)
    finishes = np.arange(segSize - 1, tsLen)
    return gu.getSumOfSquares(starts, finishes) / segSize

def getAllSumSegs(cumSums, tsLen, segSize):
    shape = np.shape(cumSums)
    if len(shape) == 1:
        return np.array([cumSums[i + segSize] - cumSums[i] for i in range(tsLen - segSize + 1)])
    else:
        return np.array([getAllSumSegs(curCumSums, tsLen, segSize) for curCumSums in cumSums])