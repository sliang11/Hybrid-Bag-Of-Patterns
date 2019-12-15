#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*-

'''
Our Hybrid-Bag-Of-Patterns (HBOP) algorithm and all its variants that are compared
in the IMPACT OF DESIGN CHOICES section of our paper.
Only the full version of HBOP is timed.

The X-means algorithm is implemented by PyClustering:

Novikov, A., 2019. PyClustering: Data Mining Library. Journal of Open Source Software, 4(36), p.1230. 
Available at: http://dx.doi.org/10.21105/joss.01230.

''' 

import sys
sys.path.append('E:/HBOP')
import os
from copy import deepcopy
import numpy as np
from Discretization import Discretizer, SLA, SAX
from TimeSeries.BagOfPatterns import BOP
from Classification.Classifier import Classifier
from Util.EvaluationUtil import chi2
import Util.BitUtil as bu
from time import perf_counter
import warnings
from sklearn.metrics import accuracy_score
import pickle
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


class HBOP(Classifier):
    
    def __init__(self, inputPath, minWinRatio = 0.025, maxWinRatio = 1, winRatioStep = 0.025, minWinLen = 10, 
                minSAXSize = 3, maxSAXSize = 8, SAXSizeStep = 1, 
                minSLASize = 6, maxSLASize = 10, SLASizeStep = 2, 
                SAXCard = 4, SLACard = 3, binSizeTh = 3, 
                maxXMeansFeat = 5000, maxClusters = 5, minClusterSize = 10):
        super().__init__(inputPath)
        self.minWinRatio = minWinRatio
        self.maxWinRatio = maxWinRatio
        self.winRatioStep = winRatioStep
        self.minSAXSize = minSAXSize
        self.maxSAXSize = maxSAXSize
        self.SAXSizeStep = SAXSizeStep
        self.minSLASize = minSLASize
        self.maxSLASize = maxSLASize
        self.SLASizeStep = SLASizeStep
        self.SAXCard = SAXCard
        self.SLACard = SLACard
        self.binSizeTh = binSizeTh
        self.maxXMeansFeat = maxXMeansFeat
        self.maxClusters = maxClusters
        self.minClusterSize = minClusterSize
    
    def createBOPId(self, winLenInd, numBitsWinLen, wordSizeInd, numBitsWordSize, wordType, numBitsWordType = 1):
        bopId = 0
        bopId = bu.appendBits(bopId, winLenInd, numBitsWinLen)
        bopId = bu.appendBits(bopId, wordSizeInd, numBitsWordSize)
        bopId = bu.appendBits(bopId, wordType, numBitsWordType)
        return bopId
    
    def createMethodId(self, wordSizeInd, numBitsWordSize, wordType, numBitsWordType):
        methodId = 0
        methodId = bu.appendBits(methodId, wordSizeInd, numBitsWordSize)
        methodId = bu.appendBits(methodId, wordType, numBitsWordType)
#         methodId = bu.appendBits(methodId, xSimId, numBitsXSimId)
        return methodId
    
    def crossValidation(self, numWords, words, wordRanks, curBopsAndBags, numBitsId, XMeansTrainLabels, 
                        XMeansNumCls, XMeansNumTrainByCls, XMeansRelabelMap = None):
        bopIds = []
        tfIdfsByCls = np.zeros((numWords, XMeansNumCls))
        meanCntsByCls = np.zeros((numWords, XMeansNumCls))
        sigmas2Centroids = np.zeros(XMeansNumCls)
        tmpSigma2Ts = np.zeros(self.numTrain)
        tmpSigma2Centroids = np.zeros((self.numTrain, XMeansNumCls))
        tmpsigmaProd = np.zeros((self.numTrain, XMeansNumCls))
        eds = np.zeros((self.numTrain, XMeansNumCls))
        
        bestAcc_ed = -1
        bestAcc_cos = -1
        
        for i in range(numWords):
            
            idx = wordRanks[i]
            word = words[idx]
            bopId = bu.getBits(word, 0, numBitsId)
            (bop, bagWord) = curBopsAndBags[bopId]
            bopIds.append(bopId)
            
            cntByCls = np.zeros(XMeansNumCls, dtype = 'uint32')
            cntByTs = np.zeros(self.numTrain, dtype = 'uint32')
            word_nid = bu.trimBits(word, numBitsId)
            for tsId, cnt in bagWord[word_nid].items(): #count by class
                label = XMeansTrainLabels[tsId]
                cntByCls[label] += cnt
                cntByTs[tsId] += cnt
            meanCntsByCls[idx][:] = cntByCls / XMeansNumTrainByCls   
            
            nonZeroCls = (np.where(cntByCls != 0))[0]
            numNonZero = len(nonZeroCls)
            idf = np.log10(1 + XMeansNumCls / numNonZero)
            tfIdfsByCls[idx][nonZeroCls] = (1 + np.log10(cntByCls[nonZeroCls])) * idf  
            sigmas2Centroids += tfIdfsByCls[idx][:] ** 2
            
            tfByTs = np.zeros(self.numTrain)
            nonZeroTs = np.where(cntByTs != 0)
            tfByTs[nonZeroTs] = 1 + np.log10(cntByTs[nonZeroTs])
            tmpSigma2Ts += tfByTs ** 2
            
            addToEd = np.zeros((self.numTrain, XMeansNumCls))
            tmpTfIdfsByCls = np.zeros((self.numTrain, XMeansNumCls))
            for j in range(self.numTrain):
                label = XMeansTrainLabels[j]
                tmpCntByCls = deepcopy(cntByCls)
                tmpNumTrainByCls = deepcopy(XMeansNumTrainByCls)
                tmpCntByCls[label] -= cntByTs[j]
                if tmpNumTrainByCls[label] > 1:
                    tmpNumTrainByCls[label] -= 1
                addToEd[j][:] = (tmpCntByCls / tmpNumTrainByCls - cntByTs[j]) ** 2
                
                nonZeroCls = (np.where(tmpCntByCls != 0))[0]
                numNonZero = len(nonZeroCls)
                if numNonZero == 0:
                    continue
                idf = np.log10(1 + XMeansNumCls / numNonZero)
                tmpTfIdfsByCls[j][nonZeroCls] = (1 + np.log10(tmpCntByCls[nonZeroCls])) * idf
            eds += addToEd
            tmpSigma2Centroids += tmpTfIdfsByCls ** 2
            tmpsigmaProd += tmpTfIdfsByCls * tfByTs[:, np.newaxis]
            
            divide = tmpSigma2Centroids * tmpSigma2Ts[:, np.newaxis]
            divide[np.where(divide == 0)] = -1
            cosSims = tmpsigmaProd ** 2 / divide
            
            preLabels_ed = np.argmin(eds, axis = 1)
            for j, preLabel in enumerate(preLabels_ed):
                preLabels_ed[j] = XMeansRelabelMap[preLabel] if XMeansRelabelMap is not None else preLabel
            acc_ed = accuracy_score(self.trainLabels, preLabels_ed)
            if acc_ed >= bestAcc_ed:
                bestAcc_ed = acc_ed
                bestPreLabels_ed = preLabels_ed
                numSelected_ed = i + 1
            preLabels_cos = np.argmax(cosSims, axis = 1)
            for j, preLabel in enumerate(preLabels_cos):
                preLabels_cos[j] = XMeansRelabelMap[preLabel] if XMeansRelabelMap is not None else preLabel
            acc_cos = accuracy_score(self.trainLabels, preLabels_cos)
            if acc_cos >= bestAcc_cos:
                bestAcc_cos = acc_cos
                bestPreLabels_cos = preLabels_cos
                bestSigmas2Centroids = deepcopy(sigmas2Centroids)
                numSelected_cos = i + 1
        
        return bopIds, bestAcc_ed, bestAcc_cos, bestPreLabels_ed, bestPreLabels_cos, \
            numSelected_ed, numSelected_cos, meanCntsByCls, tfIdfsByCls, bestSigmas2Centroids
    
    def predict(self, allBagsTs, allInfo, fineMetId, numBitsXSimId, numBitsWordType, numBitsSLAId, numBitsSAXId):
        
        methodId = bu.trimBits(fineMetId, numBitsXSimId)
        xSimId = bu.getBits(fineMetId, 0, numBitsXSimId)
        wordType = bu.getBits(methodId, 0, numBitsWordType)
        numBitsId = numBitsSLAId if wordType == 0 else numBitsSAXId
        (bops, selectedWordInfo, selectedBopIds, selectedWords, sigmas2Centroids_nx, 
            sigmas2Centroids_x, cv1_scores, XMeansRelabelMap) = allInfo[methodId]
        curBopIds = selectedBopIds[xSimId]
        curWords = selectedWords[xSimId]
        cv1_score = cv1_scores[xSimId]
        sigma2Centroids = sigmas2Centroids_nx if xSimId in (0, 1) else sigmas2Centroids_x
        XMeansRelabelMap = None if xSimId in (0, 1) else XMeansRelabelMap
        XMeansNumCls = len(XMeansRelabelMap) if XMeansRelabelMap is not None else self.numCls
        dists = np.zeros(XMeansNumCls)
        if xSimId in (1, 3):   #tfidf
            sigma2Ts = 0
            sigmaProd = np.zeros(XMeansNumCls)
            
        for word in curWords:
            infoByCls = selectedWordInfo[word][xSimId]
            word_nid = bu.trimBits(word, numBitsId)
            bopId = bu.getBits(word, 0, numBitsId)
            
            cnt = 0
            if bopId in allBagsTs.keys():
                bagTs = allBagsTs[bopId]
                if word_nid in bagTs.keys():
                    cnt = bagTs[word_nid]
            
            if xSimId in (0, 2):  #ed
                dists += (cnt - infoByCls) ** 2
            else:
                tf = 0 if cnt == 0 else 1 + np.log10(cnt)
                sigma2Ts += tf ** 2
                sigmaProd += tf * infoByCls
                    
        if xSimId in (1, 3):
            divide = sigma2Ts * sigma2Centroids
            divide[np.where(divide == 0)] = -1
            dists = 1 - sigmaProd ** 2  / divide
        preLabel = XMeansRelabelMap[np.argmin(dists)] if XMeansRelabelMap is not None else np.argmin(dists)
        return preLabel, cv1_score
        
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
        
        self.minWinLen = np.maximum(int(np.around(self.minTrainLen * self.minWinRatio)), 10, dtype = 'int32')
        self.maxWinLen = np.minimum(int(np.around(self.minTrainLen * self.maxWinRatio)), self.minTrainLen, dtype = 'int32')
        self.winLenStep = np.maximum(int(np.around(self.minTrainLen * self.winRatioStep)), 1, dtype = 'int32')
        if self.minTrainLen < self.minWinLen:
            self.minWinLen = self.minTrainLen 
        if self.minTrainLen < self.maxWinLen:
            self.maxWinLen = self.minTrainLen 
        numBitsWinLen = bu.numBits(np.ceil((self.maxWinLen - self.minWinLen) / self.winLenStep) + 1)
        numBitsWordType = 1
        numBitsXSimId = 2
        
        SLASizes = np.arange(self.maxSLASize, self.minSLASize - 1, -self.SLASizeStep, dtype = 'uint32')
        numBitsSLASize = bu.numBits(np.ceil((self.maxSLASize - self.minSLASize) / self.SLASizeStep) + 1)
        numBitsSLAId = numBitsWinLen + numBitsSLASize + numBitsWordType
        SAXSizes = np.arange(self.maxSAXSize, self.minSAXSize - 1, -self.SAXSizeStep, dtype = 'uint32')
        numBitsSAXSize = bu.numBits(np.ceil((self.maxSAXSize - self.minSAXSize) / self.SAXSizeStep) + 1)
        numBitsSAXId = numBitsWinLen + numBitsSAXSize + numBitsWordType
        
        self.XMeansTime = 0
        self.trainTime = 0
        tic = perf_counter()
        
        (allCumSums, allCumSums_2, allWCumSums) = Discretizer.getAllCumSums(trainTss_padded)
        all_cv1_scores_hy = []
        all_cv1_scoresMap_hy = {}
        self.trainTime += perf_counter() - tic
        
        all_cv1_scores = [] 
        all_cv1_scoresMap = {}
        allInfo = {} 
        transferMetIdMap = {}   
        
        for wordType in range(2):
            
            if wordType == 0:
                wordSizes = SLASizes
                numBitsId = numBitsSLAId
                numBitsWordSize = numBitsSLASize
            else:
                wordSizes = SAXSizes
                numBitsId = numBitsSAXId
                numBitsWordSize = numBitsSAXSize
            
            for wordSize in wordSizes:
                print('WordType = ', wordType, ' WordSize = ', wordSize)
                sys.stdout.flush()
                
                flg = wordType == 0 or (wordType == 1 and wordSize in {6, 7, 8})
                if flg:
                    tic = perf_counter()
                
                if wordType == 0:
                    wordSizeInd = (self.maxSLASize - wordSize) / self.SLASizeStep
                else:
                    wordSizeInd = (self.maxSAXSize - wordSize) / self.SAXSizeStep
                
                words = []
                chi2s = []
                curBopsAndBags = {}  #bopId + (bop, bagWord)
                
                for winLen in range(self.minWinLen, self.maxWinLen + 1, self.winLenStep):
                    winLenInd = (winLen - self.minWinLen) / self.winLenStep
                    
                    if wordType == 0:
                        discretizer = SLA.SLA(winLen, wordSize, self.SLACard, True, True, True, self.binSizeTh)
                        discretizedTss = discretizer.discretizeTssFromCumSums_LNR(allCumSums, allCumSums_2, allWCumSums, self.trainLens)
#                         transformedTss = discretizer.transfromTssFromCumSums(allCumSums, allCumSums_2, allWCumSums, self.trainLens)
#                         discretizedTss = discretizer.discretizeTransformedDataset_(transformedTss, self.trainLens, self.trainLabels, 'ED', 'Default')
                    else:
                        discretizer = SAX.SAX(winLen, wordSize, self.SAXCard, True, True, self.binSizeTh)
                        discretizedTss = discretizer.discretizeTssFromCumSums_LNR(allCumSums, allCumSums_2, self.trainLens)
#                         transformedTss = discretizer.transfromTssFromCumSums(allCumSums, allCumSums_2, self.trainLens)
#                         discretizedTss = discretizer.discretizeTransformedDataset_(transformedTss, self.trainLens, self.trainLabels, 'GD', 'Default')
                    
                    bop = BOP(discretizer, False)
#                         bagTss = bop.getBOP_DiscretizedTss(discretizedTss, True, -1) #With numerosity reduction, no bigrams  
                    bagWord = bop.getWordFirstBop_DiscretizedTss(discretizedTss)
                    bopId = self.createBOPId(winLenInd, numBitsWinLen, wordSizeInd, numBitsWordSize, wordType, numBitsWordType)
                    curBopsAndBags[bopId] = (bop, bagWord)      
                            
                    for word_nid, cntTs in bagWord.items():
                        feats = np.zeros(self.numTrain)
                        for tsId, cnt in cntTs.items():
                            feats[tsId] = cnt
                        chi2Val = chi2(feats, self.trainLabels)
                        word = bu.appendBits(word_nid, bopId, numBitsId)
                        words.append(word)
                        chi2s.append(chi2Val)
                
                #ranking the words
                numWords = len(words)
                wordRanks = np.argsort(-np.array(chi2s))
                
                #prepare feature matrix for xmeans clustering
                numFeats = min(numWords, self.maxXMeansFeat)
                feats = np.zeros((self.numTrain, numFeats))
                for i in range(numFeats):
                    idx = wordRanks[i]
                    word = words[idx]
                    bopId = bu.getBits(word, 0, numBitsId)
                    (bop, bagWord) = curBopsAndBags[bopId]
                    word_nid = bu.trimBits(word, numBitsId)
                    for tsId, cnt in bagWord[word_nid].items():
                        feats[tsId][i] = cnt
                
                #Xmeans
                tic_x = perf_counter()
                XMeansRelabelMap = {}   #XMeans label -> originally relabeled label
                XMeansTrainLabels = np.zeros(self.numTrain, dtype = 'uint32')
                XMeansNumTrainByCls = []
                nextXMeansLabel = 0
                for label, indsToCluster in enumerate(self.trainIndsByCls):
                    numToCluster = len(indsToCluster)
                    if numToCluster > self.minClusterSize:
                        curFeats = feats[indsToCluster][:]
                        initial_centers = kmeans_plusplus_initializer(curFeats, 2).initialize()
                        xmeans_instance = xmeans(curFeats, initial_centers, self.maxClusters)
                        xmeans_instance.process()
                        clusters = xmeans_instance.get_clusters()
                    else:
                        clusters = [range(numToCluster)]
                    for cluster in clusters:
                        XMeansTrainLabels[indsToCluster[np.array(cluster)]] = nextXMeansLabel
                        XMeansRelabelMap[nextXMeansLabel] = label
                        XMeansNumTrainByCls.append(len(cluster))
                        nextXMeansLabel += 1
                XMeansNumCls = nextXMeansLabel
                XMeansNumTrainByCls = np.array(XMeansNumTrainByCls)
                self.XMeansTime += perf_counter() - tic_x
                
                #cross validation
                bopIds, bestAcc_ed_nx, bestAcc_cos_nx, bestPreLabels_ed_nx, bestPreLabels_cos_nx, \
                numSelected_ed_nx, numSelected_cos_nx, meanCntsByCls_nx, tfIdfsByCls_nx, sigmas2Centroids_nx = \
                self.crossValidation(numWords, words, wordRanks, curBopsAndBags, numBitsId, self.trainLabels, 
                        self.numCls, self.numTrainByCls, None)
#                 matsToSave = [meanCntsByCls_nx, tfIdfsByCls_nx, sigmas2Centroids_nx]
                    
                if np.array_equal(XMeansTrainLabels, self.trainLabels):
                    bestAccs = [bestAcc_ed_nx, bestAcc_cos_nx]
                    bestPreLabels = [bestPreLabels_ed_nx, bestPreLabels_cos_nx]
                    numsSelected = np.array([numSelected_ed_nx, numSelected_cos_nx])
                else:
                    bopIds, bestAcc_ed_x, bestAcc_cos_x, bestPreLabels_ed_x, bestPreLabels_cos_x, numSelected_ed_x, numSelected_cos_x, \
                    meanCntsByCls_x, tfIdfsByCls_x, sigmas2Centroids_x = \
                    self.crossValidation(numWords, words, wordRanks, curBopsAndBags, numBitsId, XMeansTrainLabels, 
                            XMeansNumCls, XMeansNumTrainByCls, XMeansRelabelMap)
#                     matsToSave += [meanCntsByCls_x, tfIdfsByCls_x, sigmas2Centroids_x]
                    bestAccs = [bestAcc_ed_nx, bestAcc_cos_nx, bestAcc_ed_x, bestAcc_cos_x]
                    bestPreLabels = [bestPreLabels_ed_nx, bestPreLabels_cos_nx, bestPreLabels_ed_x, bestPreLabels_cos_x]
                    numsSelected = np.array([numSelected_ed_nx, numSelected_cos_nx, numSelected_ed_x, numSelected_cos_x])
                xSimIdRange = np.argsort(numsSelected)
                
                methodId = self.createMethodId(wordSizeInd, numBitsWordSize, wordType, numBitsWordType)
                numCases = len(xSimIdRange)
                cv1_scores = [None] * numCases
                selectedWords = [None] * numCases
                selectedBopIds = [None] * numCases
                curWords = set()
                curBopIds = set()
                selectedWordInfo = {}   #{word: meanCntByCls, tfIdfByCls}
                bops = {}
                prevNumSelected = 0
                for xSimId in xSimIdRange:
                    cv1_score = bestAccs[xSimId]
                    cv1_scores[xSimId] = cv1_score
                    fineMetId = bu.appendBits(methodId, xSimId, numBitsXSimId)
                    
                    if flg:
                        all_cv1_scores_hy.append(cv1_score)
                        if cv1_score not in all_cv1_scoresMap_hy.keys():
                            all_cv1_scoresMap_hy[cv1_score] = {}
                        all_cv1_scoresMap_hy[cv1_score][fineMetId] = bestPreLabels[xSimId]
                    
                    ###Not timed
                    tic_o = perf_counter()
                    all_cv1_scores.append(cv1_score)
                    if cv1_score not in all_cv1_scoresMap.keys():
                        all_cv1_scoresMap[cv1_score] = {}
                    all_cv1_scoresMap[cv1_score][fineMetId] = bestPreLabels[xSimId]
                    if numCases == 2:
                        otherMetId = bu.appendBits(methodId, xSimId + 2, numBitsXSimId)
                        transferMetIdMap[fineMetId] = otherMetId
                    time_o = perf_counter() - tic_o
                    #####################
                    
                    numSelected = numsSelected[xSimId]
                    for i in range(prevNumSelected, numSelected):
                        idx = wordRanks[i]
                        word = words[idx]
                        curWords.add(word)
                        bopId = bopIds[i]
                        if bopId not in bops.keys():
                            curBopIds.add(bopId)
                            bops[bopId] = curBopsAndBags[bopId][0]
                        if numCases == 4:
                            selectedWordInfo[word] = (meanCntsByCls_nx[idx][:], tfIdfsByCls_nx[idx][:], meanCntsByCls_x[idx][:], tfIdfsByCls_x[idx][:])
                        else:
                            selectedWordInfo[word] = (meanCntsByCls_nx[idx][:], tfIdfsByCls_nx[idx][:])
                    selectedWords[xSimId] = deepcopy(curWords)
                    selectedBopIds[xSimId] = deepcopy(curBopIds)
#                     selectedWords[xSimId] = curWords
#                     selectedBopIds[xSimId] = curBopIds
                    prevNumSelected = numSelected
                                   
                if numCases == 2:
                    XMeansRelabelMap = None
                    sigmas2Centroids_x = None
                allInfo[methodId] = (bops, selectedWordInfo, selectedBopIds, selectedWords, 
                                     sigmas2Centroids_nx, sigmas2Centroids_x, cv1_scores, XMeansRelabelMap)
                
                if flg:
                    self.trainTime += perf_counter() - tic - time_o
        
        tic = perf_counter()
        all_cv1_scores_hy = -np.sort(-np.array(all_cv1_scores_hy))
        self.fineMetIds_hy = []
        bestAcc_weighted = -1
        votesByLabels_weighted = np.zeros((self.numTrain, self.numCls))
        numMethods = 0
        prev_cv1_score = -1
        for cv1_score in all_cv1_scores_hy:
            if cv1_score == prev_cv1_score:
                continue
            prev_cv1_score = cv1_score
            metPreMap = all_cv1_scoresMap_hy[cv1_score]
            for fineMetId, preLabels_met in metPreMap.items():
                self.fineMetIds_hy.append(fineMetId)
                numMethods += 1
                
                for j in range(self.numTrain):
                    votesByLabels_weighted[j][preLabels_met[j]] += cv1_score
                preLabels_weighted = np.argmax(votesByLabels_weighted, axis = 1)
                acc_weighted = accuracy_score(self.trainLabels, preLabels_weighted)
                if acc_weighted >= bestAcc_weighted:
                    bestAcc_weighted = acc_weighted
                    bestNumMethods_weighted_hy = numMethods
        self.fineMetIds_hy = self.fineMetIds_hy[: bestNumMethods_weighted_hy]
        self.allInfo_hy = {}
        for fineMetId in self.fineMetIds_hy:
            methodId = bu.trimBits(fineMetId, numBitsXSimId)
            if methodId not in self.allInfo_hy.keys():
                self.allInfo_hy[methodId] = allInfo[methodId]  
        self.trainTime += perf_counter() - tic
            
        ####Not timed
        all_cv1_scores = -np.sort(-np.array(all_cv1_scores))
        allValidWordTypes = [[0, 1], [1]]
        allValidSLAWordSizes = [[6, 8, 10], []]
        allValidSAXWordSizes = [[6, 7, 8], [3, 4, 5, 6, 7, 8]]
        allValidXSimIds = [[0, 1, 2, 3], [0, 1], [2, 3]]
        self.allFineMetIds = []
        self.allBestNumMethods = []
        self.allBestNumMethods_weighted = []
        for i, validWordTypes in enumerate(allValidWordTypes):
            validSLAWordSizes = np.array(allValidSLAWordSizes[i], dtype = 'uint32')
            validSLAWordSizeInds = (self.maxSLASize - validSLAWordSizes) / self.SLASizeStep
            validSAXWordSizes = np.array(allValidSAXWordSizes[i], dtype = 'uint32')
            validSAXWordSizeInds = (self.maxSAXSize - validSAXWordSizes) / self.SAXSizeStep
            for validXSimIds in allValidXSimIds:
                
                bestAcc = -1
                bestAcc_weighted = -1
                votesByLabels = np.zeros((self.numTrain, self.numCls))
                votesByLabels_weighted = np.zeros((self.numTrain, self.numCls))
                fineMetIds = []
                numMethods = 0
                prev_cv1_score = -1
                for cv1_score in all_cv1_scores:
                    if cv1_score == prev_cv1_score:
                        continue
                    prev_cv1_score = cv1_score
                    metPreMap = all_cv1_scoresMap[cv1_score]
                    for fineMetId, preLabels_met in metPreMap.items():
                        
                        wordType = bu.getBits(fineMetId, numBitsXSimId, numBitsWordType)
                        if wordType not in validWordTypes:
                            continue
                        
                        if wordType == 0:
                            numBitsWordSize = numBitsSLASize
                        elif wordType == 1:
                            numBitsWordSize = numBitsSAXSize
                        
                        wordSizeInd = bu.getBits(fineMetId, numBitsXSimId + numBitsWordType, numBitsWordSize)
                        if wordType == 0 and wordSizeInd not in validSLAWordSizeInds:
                            continue
                        if wordType == 1 and wordSizeInd not in validSAXWordSizeInds:
                            continue
                        
                        xSimId = bu.getBits(fineMetId, 0, numBitsXSimId)
                        if xSimId not in validXSimIds:
                            if fineMetId not in transferMetIdMap.keys():
                                continue
                            else:
                                otherMethodId = transferMetIdMap[fineMetId]
                                if bu.getBits(otherMethodId, 0, numBitsXSimId) not in validXSimIds:
                                    continue
                        fineMetIds.append(fineMetId)
                        numMethods += 1
                        
                        for j in range(self.numTrain):
                            votesByLabels[j][preLabels_met[j]] += 1
                            votesByLabels_weighted[j][preLabels_met[j]] += cv1_score
                        preLabels = np.argmax(votesByLabels, axis = 1)
                        preLabels_weighted = np.argmax(votesByLabels_weighted, axis = 1)
                        acc = accuracy_score(self.trainLabels, preLabels)
                        acc_weighted = accuracy_score(self.trainLabels, preLabels_weighted)
                        if acc >= bestAcc:
                            bestAcc = acc
                            bestNumMethods = numMethods
                        if acc_weighted >= bestAcc_weighted:
                            bestAcc_weighted = acc_weighted
                            bestNumMethods_weighted = numMethods
                fineMetIds = fineMetIds[: max([bestNumMethods, bestNumMethods_weighted])]            
                self.allFineMetIds.append(fineMetIds)
                self.allBestNumMethods.append(bestNumMethods)
                self.allBestNumMethods_weighted.append(bestNumMethods_weighted)
        
        self.allInfo = {}
        for fineMetIds in self.allFineMetIds:
            for fineMetId in fineMetIds:
                methodId = bu.trimBits(fineMetId, numBitsXSimId)
                if methodId not in self.allInfo.keys():
                    self.allInfo[methodId] = allInfo[methodId]  
        ############################
            
    
    def test(self):
        
        numCases = len(self.allFineMetIds)
        numCases *= 2
        self.preLabels_hy = np.zeros(self.numTest, dtype = 'uint32')    
        self.preLabels = np.zeros((numCases, self.numTest), dtype = 'uint32')  
        
        numBitsWinLen = bu.numBits(np.ceil((self.maxWinLen - self.minWinLen) / self.winLenStep) + 1)
        numBitsWordType = 1
        numBitsXSimId = 2
        
        numBitsSLASize = bu.numBits(np.ceil((self.maxSLASize - self.minSLASize) / self.SLASizeStep) + 1)
        numBitsSLAId = numBitsWinLen + numBitsSLASize + numBitsWordType
        numBitsSAXSize = bu.numBits(np.ceil((self.maxSAXSize - self.minSAXSize) / self.SAXSizeStep) + 1)
        numBitsSAXId = numBitsWinLen + numBitsSAXSize + numBitsWordType
        
        self.testTimePerTs = 0
        
        for tsId in range(len(self.testTss)):
            
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
            
            (cumSums, cumSums_2, wCumSums) = Discretizer.getAllCumSums(ts)
            allBagsTs = {}
            for methodId, (bops, selectedWordInfo, selectedBopIds, selectedWords, sigmas2Centroids_nx, 
                           sigmas2Centroids_x, cv1_scores, XMeansRelabelMap) in self.allInfo_hy.items():
                
                wordType = bu.getBits(methodId, 0, numBitsWordType)
                for bopId, bop in bops.items():
                    if bopId in allBagsTs.keys():
                        continue
                    
                    if bop.discretizer.winLen > tsLen:
                        curCumSums = np.concatenate((cumSums, cumSums[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                        curCumSums_2 = np.concatenate((cumSums_2, cumSums_2[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                        curWCumSums = np.concatenate((wCumSums, wCumSums[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                        curTsLen = bop.discretizer.winLen
                    else:
                        curCumSums = cumSums
                        curCumSums_2 = cumSums_2
                        curWCumSums = wCumSums
                        curTsLen = tsLen
                        
                    if wordType == 0:
#                         transformedTs = bop.discretizer.transformTsFromCumSums(cumSums, cumSums_2, wCumSums)
#                         discretizedTs = bop.discretizer.discretizeTransformedTs(transformedTs)
                        discretizedTs = bop.discretizer.discretizeTsFromCumSums_LNR(curCumSums, curCumSums_2, curWCumSums, curTsLen)
                        bagTs = bop.getBOP_DiscretizedTs(discretizedTs)
                    else:
#                         transformedTs = bop.discretizer.transformTsFromCumSums(cumSums, cumSums_2)
#                         discretizedTs = bop.discretizer.discretizeTransformedTs(transformedTs)
                        discretizedTs = bop.discretizer.discretizeTsFromCumSums_LNR(curCumSums, curCumSums_2, curTsLen)
                        bagTs = bop.getBOP_DiscretizedTs(discretizedTs)
                    allBagsTs[bopId] = bagTs
            
            votesByLabel_weighted = np.zeros(self.numCls)
            for fineMetId in self.fineMetIds_hy:
                preLabel, cv1_score = self.predict(allBagsTs, self.allInfo_hy, fineMetId, 
                            numBitsXSimId, numBitsWordType, numBitsSLAId, numBitsSAXId)
                votesByLabel_weighted[preLabel] += cv1_score
            self.preLabels_hy[tsId] = np.argmax(votesByLabel_weighted)
            self.testTimePerTs += perf_counter() - tic
            
            ###########Not timed
            for methodId, (bops, selectedWordInfo, selectedBopIds, selectedWords, sigmas2Centroids_nx, 
                           sigmas2Centroids_x, cv1_scores, XMeansRelabelMap) in self.allInfo.items():
                
                wordType = bu.getBits(methodId, 0, numBitsWordType)
                for bopId, bop in bops.items():
                    if bopId in allBagsTs.keys():
                        continue
                    
                    if bop.discretizer.winLen > tsLen:
                        curCumSums = np.concatenate((cumSums, cumSums[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                        curCumSums_2 = np.concatenate((cumSums_2, cumSums_2[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                        curWCumSums = np.concatenate((wCumSums, wCumSums[-1] * np.ones(bop.discretizer.winLen - tsLen)))
                        curTsLen = bop.discretizer.winLen
                    else:
                        curCumSums = cumSums
                        curCumSums_2 = cumSums_2
                        curWCumSums = wCumSums
                        curTsLen = tsLen
                        
                    if wordType == 0:
#                         transformedTs = bop.discretizer.transformTsFromCumSums(cumSums, cumSums_2, wCumSums)
#                         discretizedTs = bop.discretizer.discretizeTransformedTs(transformedTs)
                        discretizedTs = bop.discretizer.discretizeTsFromCumSums_LNR(curCumSums, curCumSums_2, curWCumSums, curTsLen)
                        bagTs = bop.getBOP_DiscretizedTs(discretizedTs)
                    else:
#                         transformedTs = bop.discretizer.transformTsFromCumSums(cumSums, cumSums_2)
#                         discretizedTs = bop.discretizer.discretizeTransformedTs(transformedTs)
                        discretizedTs = bop.discretizer.discretizeTsFromCumSums_LNR(curCumSums, curCumSums_2, curTsLen)
                        bagTs = bop.getBOP_DiscretizedTs(discretizedTs)
                    allBagsTs[bopId] = bagTs
            
            for case, fineMetIds in enumerate(self.allFineMetIds):
                bestNumMethods = self.allBestNumMethods[case]
                bestNumMethods_weighted = self.allBestNumMethods_weighted[case]
                votesByLabel = np.zeros(self.numCls)
                votesByLabel_weighted = np.zeros(self.numCls)
                for i, fineMetId in enumerate(fineMetIds):
                    preLabel, cv1_score = self.predict(allBagsTs, self.allInfo, fineMetId, 
                            numBitsXSimId, numBitsWordType, numBitsSLAId, numBitsSAXId)
                    if i < bestNumMethods:
                        votesByLabel[preLabel] += 1
                    if i < bestNumMethods_weighted:
                        votesByLabel_weighted[preLabel] += cv1_score
                self.preLabels[case][tsId] = np.argmax(votesByLabel)
                self.preLabels[case + len(self.allFineMetIds)][tsId] = np.argmax(votesByLabel_weighted)
        
        self.testTimePerTs /= self.numTest
        self.accuracy = accuracy_score(self.testLabels, self.preLabels_hy)
        self.accuracies = np.zeros(numCases)
        for i in range(numCases):
            self.accuracies[i] = accuracy_score(self.testLabels, self.preLabels[i][:])
        
if __name__ == '__main__':
#     warnings.filterwarnings("ignore")
    
    dataset = sys.argv[1]
    runId = sys.argv[2]
    inputPath = sys.argv[3]
    savePath = sys.argv[4]
    
    hbop = HBOP(inputPath = inputPath)
    hbop.loadUCRDataset_2018(dataset, 'TRAIN')
    hbop.train()
    hbop.loadUCRDataset_2018(dataset, 'TEST')
    hbop.test()
    
    fName = savePath + '/accuracies_' + dataset + '_HBOP.' + runId + '.txt'
    file = open(fName, 'w')
    for i in range(len(hbop.accuracies)): 
        file.write(str(hbop.accuracies[i]) + "\n")
    file.close()
    
    fName = savePath + '/time_'+ dataset + '_HBOP.' + runId + '.txt'
    file = open(fName, 'w')
    file.write(dataset + "    " + str(hbop.trainTime) + "   " + str(hbop.testTimePerTs) + "    " + str(hbop.accuracy) + "\n")
