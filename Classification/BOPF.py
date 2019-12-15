#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 
'''
The Bag-Of-Pattern-Features (BOPF) algorithm

Xiaosheng Li, Jessica Lin:
Linear Time Complexity Time Series Classification with Bag-of-Pattern-Features. ICDM 2017: 277-286

'''

import sys
sys.path.append('E:/HBOP')

from copy import deepcopy
import numpy as np
from Discretization import SAX
from TimeSeries.BagOfPatterns import BOP
from Classification.Classifier import Classifier
from Util.EvaluationUtil import FStat_2
# from sklearn.preprocessing import scale
from Util import GeneralUtil as gu
from time import perf_counter
import warnings
from sklearn.metrics import accuracy_score
import pickle
import os


class BOPF(Classifier):
    
    def __init__(self, inputPath, minWinRatio = 0.025, maxWinRatio = 1, winRatioStep = 0.025, minWinLen = 10, 
                minWordSize = 3, maxWordSize = 7, wordSizeStep = 1, card = 4, topK = 30, accRatio = 0.7, binSizeTh = 3):
        super().__init__(inputPath)
        self.minWinRatio = minWinRatio
        self.maxWinRatio = maxWinRatio
        self.winRatioStep = winRatioStep
        self.minWinLen = minWinLen
        self.minWordSize = minWordSize
        self.maxWordSize = maxWordSize
        self.wordSizeStep = wordSizeStep
        self.card = card
        self.topK = topK
        self.accRatio = accRatio
        self.binSizeTh = binSizeTh
    
    def crossValidation(self, numWords, words, wordRanks, bagWord):
        tfIdfsByCls = np.zeros((numWords, self.numCls))
        meanCntsByCls = np.zeros((numWords, self.numCls))
        sigmas2Centroids = np.zeros(self.numCls)
        tmpSigma2Ts = np.zeros(self.numTrain)
        tmpSigma2Centroids = np.zeros((self.numTrain, self.numCls))
        tmpsigmaProd = np.zeros((self.numTrain, self.numCls))
        eds = np.zeros((self.numTrain, self.numCls))
        
        bestAcc_ed = -1
        bestAcc_cos = -1
        
        for i in range(numWords):
            idx = wordRanks[i]
            word = words[idx]
            
            cntByCls = np.zeros(self.numCls, dtype = 'uint32')
            cntByTs = np.zeros(self.numTrain, dtype = 'uint32')
            for tsId, cnt in bagWord[word].items(): #count by class
                label = self.trainLabels[tsId]
                cntByCls[label] += cnt
                cntByTs[tsId] += cnt
            meanCntsByCls[idx][:] = cntByCls / self.numTrainByCls
            
            nonZeroCls = (np.where(cntByCls != 0))[0]
            numNonZero = len(nonZeroCls)
            idf = np.log10(1 + self.numCls / numNonZero)
            tfIdfsByCls[idx][nonZeroCls] = (1 + np.log10(cntByCls[nonZeroCls])) * idf
            sigmas2Centroids += tfIdfsByCls[idx][:] ** 2
            
            tfByTs = np.zeros(self.numTrain)
            nonZeroTs = np.where(cntByTs != 0)
            tfByTs[nonZeroTs] = 1 + np.log10(cntByTs[nonZeroTs])
            tmpSigma2Ts += tfByTs ** 2
            
            addToEd = np.zeros((self.numTrain, self.numCls))
            tmpTfIdfsByCls = np.zeros((self.numTrain, self.numCls))
            for j in range(self.numTrain):
                label = self.trainLabels[j]
                tmpCntByCls = deepcopy(cntByCls)
                tmpNumTrainByCls = deepcopy(self.numTrainByCls)
                tmpCntByCls[label] -= cntByTs[j]
                if tmpNumTrainByCls[label] > 1:
                    tmpNumTrainByCls[label] -= 1
                addToEd[j][:] = (tmpCntByCls / tmpNumTrainByCls - cntByTs[j]) ** 2
                
                nonZeroCls = (np.where(tmpCntByCls != 0))[0]
                numNonZero = len(nonZeroCls)
                if numNonZero == 0:
                    continue
                idf = np.log10(1 + self.numCls / numNonZero)
                tmpTfIdfsByCls[j][nonZeroCls] = (1 + np.log10(tmpCntByCls[nonZeroCls])) * idf
            eds += addToEd
            tmpSigma2Centroids += tmpTfIdfsByCls ** 2
            tmpsigmaProd += tmpTfIdfsByCls * tfByTs[:, np.newaxis]
            
            divide = tmpSigma2Centroids * tmpSigma2Ts[:, np.newaxis]
            divide[np.where(divide == 0)] = -1
            cosSims = tmpsigmaProd ** 2 / divide
            
            preLabels_ed = np.argmin(eds, axis = 1)
            for j, preLabel in enumerate(preLabels_ed):
                preLabels_ed[j] = preLabel
            acc_ed = accuracy_score(self.trainLabels, preLabels_ed)
            if acc_ed >= bestAcc_ed:
                bestAcc_ed = acc_ed
                numSelected_ed = i + 1
            preLabels_cos = np.argmax(cosSims, axis = 1)
            for j, preLabel in enumerate(preLabels_cos):
                preLabels_cos[j] = preLabel
            acc_cos = accuracy_score(self.trainLabels, preLabels_cos)
            if acc_cos >= bestAcc_cos:
                bestAcc_cos = acc_cos
                bestSigmas2Centroids = deepcopy(sigmas2Centroids)
                numSelected_cos = i + 1
        return bestAcc_ed, bestAcc_cos, numSelected_ed, numSelected_cos, meanCntsByCls, tfIdfsByCls, bestSigmas2Centroids
    
    def train(self):
        
        trainTss_padded = []
        maxTsLen = max(self.trainLens)
        for i in range(self.numTrain):
            ts = np.array(self.trainTss[i])
            tsLen = self.trainLens[i]
#             zTs = scale(np.array(ts))
#             zTs = np.concatenate((zTs, np.zeros(maxTsLen - tsLen)))
#             trainTss_padded.append(zTs)
            ts = np.concatenate((ts, np.zeros(maxTsLen - tsLen)))
            trainTss_padded.append(ts)
        trainTss_padded = np.array(trainTss_padded)
        
        self.minWinLen = np.maximum(int(np.around(self.minTrainLen * self.minWinRatio)), self.minWinLen, dtype = 'int32')
        self.maxWinLen = np.minimum(int(np.around(self.minTrainLen * self.maxWinRatio)), self.minTrainLen, dtype = 'int32')
        self.winLenStep = np.maximum(int(np.around(self.minTrainLen * self.winRatioStep)), 1, dtype = 'int32')
        if self.minTrainLen < self.minWinLen:
            self.minWinLen = self.minTrainLen 
        if self.minTrainLen < self.maxWinLen:
            self.maxWinLen = self.minTrainLen 
#         numBitsWinLen = bu.numBits(np.ceil((self.maxWinLen - self.minWinLen) / self.winLenStep) + 1)
#         numBitsWordSize = bu.numBits(np.ceil((self.maxWordSize - self.minWordSize) / self.wordSizeStep) + 1)
        
        tic = perf_counter()
        
        allCumSums = gu.getCumSums(trainTss_padded)
        allCumSums_2 = gu.getCumSums(trainTss_padded * trainTss_padded)
        
        all_cv1_scores = [[], []]
#         allMethodIds = [[], []] 
        allInfo = []
        for wordSize in range(self.minWordSize, self.maxWordSize + 1, self.wordSizeStep):  
            for winLen in range(self.minWinLen, self.maxWinLen + 1, self.winLenStep):

                discretizer = SAX.SAX(winLen, wordSize, self.card, True, True, self.binSizeTh)
                transformedTss = discretizer.transfromTssFromCumSums(allCumSums, allCumSums_2, self.trainLens)
                discretizedTss = discretizer.discretizeTransformedDataset_(transformedTss, self.trainLens, None, 'GD', 'Default')
                bop = BOP(discretizer, False)
                bagWord = bop.getWordFirstBop_DiscretizedTss(discretizedTss)
                
                words = []
                fs = []
                for word, cntTs in bagWord.items():
                    feats = np.zeros(self.numTrain)
                    for tsId, cnt in cntTs.items():
                        feats[tsId] = cnt
                    f = FStat_2(feats, self.trainLabels, self.numCls)
                    if f:
                        words.append(word)
                        fs.append(f)
            
                numWords = len(words)
                if numWords == 0:
                    continue
                wordRanks = np.argsort(-np.array(fs))
                
                bestAcc_ed, bestAcc_cos, numSelected_ed, numSelected_cos, meanCntsByCls, tfIdfsByCls, sigmas2Centroids\
                 = self.crossValidation(numWords, words, wordRanks, bagWord)
                
                bestAccs = [bestAcc_ed, bestAcc_cos]
                numsSelected = np.array([numSelected_ed, numSelected_cos])
                simIdRange = np.argsort(numsSelected)
                selectedWordInfo = {}
                selectedWords = [None, None]
                
#                 methodId = self.createMethodId(winLenInd, numBitsWinLen, wordSizeInd, numBitsWordSize)
                prevNumSelected = 0
                curSelectedWords = set()
                for simId in simIdRange:
                    cv1_score = bestAccs[simId]
                    all_cv1_scores[simId].append(cv1_score)
#                     allMethodIds[simId].append(methodId)
                    
                    numSelected = numsSelected[simId]
                    for i in range(prevNumSelected, numSelected):
                        idx = wordRanks[i]
                        word = words[idx]
                        curSelectedWords.add(word)
                        selectedWordInfo[word] = (meanCntsByCls[idx][:], tfIdfsByCls[idx][:])
                    selectedWords[simId] = deepcopy(curSelectedWords)
                    prevNumSelected = numSelected
                allInfo.append((bop, selectedWords, selectedWordInfo, sigmas2Centroids))
        
        self.allMethodIds = []
        allAvgAcc = np.empty(2)
        for i in range(2):
            cur_cv1_scores = np.array(all_cv1_scores[i])
            numMet = len(cur_cv1_scores)
            if numMet > self.topK:
                methodIds = np.argpartition(-cur_cv1_scores, self.topK)[: self.topK]
            else:
                methodIds = np.arange(numMet)
            self.allMethodIds.append(set(methodIds))
            allAvgAcc[i] = np.mean(cur_cv1_scores[methodIds])
        maxAcc = np.amax(allAvgAcc)
        for i in range(2):
            if allAvgAcc[i] <= self.accRatio * maxAcc:
                self.allMethodIds[i] = set()
        
        self.allInfo = {}
        for methodIds in self.allMethodIds:
            for methodId in methodIds:
                if methodId not in self.allInfo.keys():
                    self.allInfo[methodId] = allInfo[methodId]   
            
        toc = perf_counter()
        self.trainTime = toc - tic
            
    
    def test(self):
        
        self.testTimePerTs = 0
        
        self.preLabels = np.zeros(self.numTest, dtype = 'uint32')
        for tsId in range(self.numTest):
            if int(tsId) % 10 == 0:
                print(tsId, end = ', ')
                sys.stdout.flush()
            if int(tsId) % 100 == 0:
                print()
                sys.stdout.flush()
                
#             ts = scale(self.testTss[tsId])
            ts = np.array(self.testTss[tsId])
            tsLen = self.testLens[tsId]
            
            tic = perf_counter()
            
            cumSums = gu.getCumSums(ts)
            cumSums_2 = gu.getCumSums(ts * ts)
            votes = np.zeros(self.numCls, dtype = 'uint32')
            for methodId, (bop, selectedWords, selectedWordInfo, sigma2Centroids) in self.allInfo.items():

                if bop.discretizer.winLen > tsLen:
                    curCumSums = np.concatenate((cumSums, cumSums[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                    curCumSums_2 = np.concatenate((cumSums_2, cumSums_2[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                else:
                    curCumSums = cumSums
                    curCumSums_2 = cumSums_2
                    
                transformedTs = bop.discretizer.transformTsFromCumSums(curCumSums, curCumSums_2)
                discretizedTs = bop.discretizer.discretizeTransformedTs(transformedTs)
                bagTs = bop.getBOP_DiscretizedTs(discretizedTs)
                
                for simId, methodIds in enumerate(self.allMethodIds):
                    if methodId not in methodIds:
                        continue
                    curSelectedWords = selectedWords[simId]
                    
                    dists = np.zeros(self.numCls)
                    if simId == 1:
                        sigma2Ts = 0
                        sigmaProd = np.zeros(self.numCls)
                    for word in curSelectedWords:
                        infoByCls = selectedWordInfo[word][simId]
                        
                        cnt = 0
                        if word in bagTs.keys():
                            cnt = bagTs[word]
                        
                        if simId == 0:  #ed
                            dists += (cnt - infoByCls) ** 2
                        else:
                            tf = 0 if cnt == 0 else 1 + np.log10(cnt)
                            sigma2Ts += tf ** 2
                            sigmaProd += tf * infoByCls
                                
                    if simId == 1:
                        divide = sigma2Ts * sigma2Centroids
                        divide[np.where(divide == 0)] = -1
                        dists = 1 - sigmaProd ** 2  / divide
                    preLabel = np.argmin(dists)
                    votes[preLabel] += 1
            
            self.preLabels[tsId] = np.argmax(votes) 
            
            toc = perf_counter()
            self.testTimePerTs += toc - tic
            
        self.accuracy = accuracy_score(self.testLabels, self.preLabels)
        self.testTimePerTs /= self.numTest
        
if __name__ == '__main__':
#     warnings.filterwarnings("ignore")
    
    dataset = sys.argv[1]
    runId = sys.argv[2]
    inputPath = sys.argv[3]
    savePath = sys.argv[4]
    
    bopf = BOPF(inputPath = inputPath)
    bopf.loadUCRDataset_2018(dataset, 'TRAIN')
    bopf.train()
    bopf.loadUCRDataset_2018(dataset, 'TEST')
    bopf.test()
    
    fName = savePath + '/accuracy_' + dataset + '_BOPF_' + runId + '.txt'
    file = open(fName, 'w')
    file.write(str(bopf.accuracy))
    file.close()
    
    fName = savePath + '/time_'+ dataset + '_BOPF_' + runId + '.txt'
    file = open(fName, 'w')
    file.write(dataset + "    " + str(bopf.trainTime) + "   " + str(bopf.testTimePerTs) + "\n")
    
