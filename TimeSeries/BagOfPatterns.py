#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 

'''
Functions with regard to the Bag-of-Patterns representation
'''
import numpy as np
from Discretization.Discretizer import Discretizer
import Util.BitUtil as bu
from copy import deepcopy

class BOP(object):
    
    def __init__(self, discretizer, binary):
        
        self.discretizer = deepcopy(discretizer)
        self.binary = binary
    
    def getBOP_DiscretizedTs(self, discretizedTs, numReduct = True, indictBi = -1): #numerosity reduction optional, include bigram optional.
        '''
        indictBi: -1: No bigrams. 0: Current words are unigrams, but we want bigrams latter. \
        1: Current words are bigrams, and we do not need other words.
        
        The idea of using bigrams comes from 
        
        Patrick Schafer, Ulf Leser:
        Fast and Accurate Time Series Classification with WEASEL. CIKM 2017: 637-646
        
        This idea is not used in the final version of HBOP for efficiency concerns.
        
        '''
        #unigram
        if self.binary:
            bagTs = np.unique(discretizedTs)
            if indictBi >= 0:
                for i in range(len(bagTs)):
                    bagTs[i] = bu.appendBits(bagTs[i], indictBi, 1)
            bagTs = set(bagTs)
        else:
            bagTs = {}
            if numReduct:
                prevWord = None
            for word in discretizedTs:
                if not numReduct or word != prevWord:
                    if numReduct:
                        prevWord = word
                    if indictBi >= 0:
                        word = bu.appendBits(word, indictBi, 1)
                    bagTs[word] = bagTs[word] + 1 if word in bagTs.keys() else 1            
        
        #bigram            
        if indictBi == 0:
            numBigrams = len(discretizedTs) - self.discretizer.winLen
            if numBigrams > 0:
                bigrams = np.empty(numBigrams, dtype = 'int64')
                shift = bu.numBits(self.discretizer.card) * self.discretizer.wordSize
                for i in range(numBigrams):
                    bigrams[i] = bu.appendBits(discretizedTs[i], discretizedTs[i + self.discretizer.winLen], shift)
                biBagTs = self.getBOP_DiscretizedTs(bigrams, numReduct, 1)
                if self.binary:
                    bagTs = bagTs | biBagTs
                else:
                    bagTs.update(biBagTs)
        return bagTs
    
    def getBOP_DiscretizedTss(self, discretizedTss, numReduct = True, indictBi = -1):
        return [self.getBOP_DiscretizedTs(discretizedTs, numReduct, indictBi) for discretizedTs in discretizedTss]
    
    def getWordFirstBop_DiscretizedTss(self, discretizedTss, numReduct = True, indictBi = -1):
        '''
        indictBi: -1: No bigrams. 0: Current words are unigrams, but we want bigrams latter. \
        1: Current words are bigrams, and we do not need other words.
        
        The idea of using bigrams comes from 
        
        Patrick Schafer, Ulf Leser:
        Fast and Accurate Time Series Classification with WEASEL. CIKM 2017: 637-646
        
        This idea is not used in the final version of HBOP for efficiency concerns.
        
        '''
        bagWord = {}
        allBigrams = []
        for i, discretizedTs in enumerate(discretizedTss):
            if self.binary:
                bagTs = np.unique(discretizedTs)
                for word in bagTs:
                    if indictBi >= 0:
                        word = bu.appendBits(word, indictBi, 1)
                    if word in bagWord.keys():
                        bagWord[word].add(i)
                    else:
                        bagWord[word] = {i}
                
            else:
                if numReduct:
                    prevWord = None
                for word in discretizedTs:
                    if not numReduct or word != prevWord:
                        if numReduct:
                            prevWord = word
                        if indictBi >= 0:
                            word = bu.appendBits(word, indictBi, 1)
                        if word in bagWord.keys():
                            bagWord[word][i] = bagWord[word][i] + 1 if i in bagWord[word].keys() else 1
                        else:
                            bagWord[word] = {i: 1}            
                                      
            if indictBi == 0:
                numBigrams = len(discretizedTs) - self.discretizer.winLen
                if numBigrams > 0:
                    bigrams = np.empty(numBigrams, dtype = 'int64')
                    shift = bu.numBits(self.discretizer.card) * self.discretizer.wordSize
                    for j in range(numBigrams):
                        bigrams[j] = bu.appendBits(discretizedTs[j], discretizedTs[j + self.discretizer.winLen], shift)
                    allBigrams.append(bigrams)
        if len(allBigrams) > 0:
            bagBigrams = self.getWordFirstBop_DiscretizedTss(allBigrams, numReduct, 1)
            bagWord.update(bagBigrams)
        return bagWord
            

    def getBOP_Ts(self, ts, numReduct = True, indictBi = -1):
        discretizedTs = self.discretizer.discretizeTs(ts)
        return self.getBOP_DiscretizedTs(discretizedTs, numReduct, indictBi)
    
    def getBOP_Dataset(self, tss, labels = None, boundStrategy = 'ED', orderStrategy = 'Default', numReduct = True, indictBi = -1):
        discretizedTss = self.discretizer.discretizeDataset(tss, labels, boundStrategy, orderStrategy)
        return [self.getBOP_DiscretizedTs(discretizedTs, numReduct, indictBi) for discretizedTs in discretizedTss]
    
    def getBOPFromBOP_Class(self, bagTss, labels):
        
#         bagTss = self.getBOP_Dataset(tss, labels, boundStrategy, orderStrategy)
        bagCls = {}
        for i in range(len(labels)):
            bagTs = bagTss[i]
            label = labels[i]
            
            if self.binary:
                for word in bagTs:
                    if word not in bagCls.keys():
                        bagCls[word] = set()
                    bagCls[word].add(label)
            else:
                for word, cnt in bagTs.items(): 
                    if word not in bagCls.keys():
                        bagCls[word] = {}
                    if label in bagCls[word].keys():
                        bagCls[word][label] += cnt
                    else:
                        bagCls[word][label] = cnt
        return bagCls
    
    def removeFromBOP_Class(self, bagCls, bagTss, labels, idx):
        newBagCls = deepcopy(bagCls)
        bagTs = bagTss[idx]
        label = labels[idx]
        if self.binary:
            for word in bagTs:
                newBagCls[word].remove[label]
                if not newBagCls[word]:
                    del newBagCls[word]
        else:
            for word, cnt in bagTs.items():
                newBagCls[word][label] -= cnt
                if not newBagCls[word][label]:
                    del newBagCls[word][label]
                if not newBagCls[word]:
                    del newBagCls[word]
        return newBagCls
    
    def increGetBOP_Ts(self, prevBagTs, numLetters = 2, indictBi = -1):
        
        '''
        The idea of incrementally obtain bag-of-patterns comes from 
        
        Patrick Schafer:
        The BOSS is concerned with time series classification in the presence of noise. Data Min. Knowl. Discov. 29(6): 1505-1530 (2015)
        
        This method is suitable to SFA, not to SLA and SAX. We have not used this.
        
        '''
        
        numBitsTrim = bu.numBits(self.discretizer.card) * numLetters
#         if self.binary:
        curBagTs = set() if self.binary else {}
        tmp_prev = prevBagTs if self.binary else prevBagTs.keys()
        for word in tmp_prev:
            if indictBi >= 0:
                isBigram = bu.getBits(word, 0, 1)
                if isBigram:
                    numBitsUni = bu.numBits(self.discretizer.card) * self.discretizer.wordSize
                    newUniWord_0 = bu.trimBits(word, numBitsTrim + numBitsUni + 1)
                    newUniWord_1 = bu.getBits(word, numBitsTrim + 1, numBitsUni - numBitsTrim)
                    newWord = bu.appendBits(newUniWord_0, newUniWord_1, numBitsUni - numBitsTrim)
                    newWord = bu.appendBits(newWord, isBigram, 1)
                    tmp_cur = curBagTs if self.binary else curBagTs.keys()
                    if newWord not in tmp_cur:
                        if not self.binary:
                            curBagTs[newWord] = 0
                        breakFlg = False
                        for i in range(2 ** numBitsTrim):
                            affUniword_0 = bu.appendBits(newUniWord_0, i, numBitsTrim)
                            for j in range(2 ** numBitsTrim):
                                affUniword_1 = bu.appendBits(newUniWord_1, j, numBitsTrim)
                                affWord = bu.appendBits(affUniword_0, affUniword_1, numBitsUni)
                                affWord = bu.appendBits(affWord, isBigram, 1)
                                if affWord in tmp_prev:
                                    if self.binary:
                                        curBagTs.add(newWord)
                                        breakFlg = True
                                        break
                                    else:
                                        curBagTs[newWord] += prevBagTs[affWord]
                            if breakFlg:
                                break
                else:
                    newWord_ni = bu.trimBits(word, numBitsTrim + 1)
                    newWord = bu.appendBits(newWord_ni, isBigram, 1)
                    tmp_cur = curBagTs if self.binary else curBagTs.keys()
                    if newWord not in tmp_cur:
                        if not self.binary:
                            curBagTs[newWord] = 0
                        for i in range(2 ** numBitsTrim):
                            affWord = bu.appendBits(newWord_ni, i, numBitsTrim)
                            affWord = bu.appendBits(affWord, isBigram, 1)
                            if affWord in tmp_prev:
                                if self.binary:
                                    curBagTs.add(newWord)
                                    break
                                else:
                                    curBagTs[newWord] += prevBagTs[affWord]
            else:
                newWord = bu.trimBits(word, numBitsTrim)
                tmp_cur = curBagTs if self.binary else curBagTs.keys()
                if newWord not in tmp_cur:
                    if not self.binary:
                        curBagTs[newWord] = 0
                    for i in range(2 ** numBitsTrim):
                        affWord = bu.appendBits(newWord, i, numBitsTrim)
                        if affWord in tmp_prev:
                            if self.binary:
                                curBagTs.add(newWord)
                                break
                            else:
                                curBagTs[newWord] += prevBagTs[affWord]
        return curBagTs                            
        
    
    def increGetBOP_Tss(self, prevBagTss, numLetters = 2, indictBi = -1):
        return [self.increGetBOP_Ts(prevBagTs, numLetters, indictBi) for prevBagTs in prevBagTss]
    
    def increGetWordFirstBop_Tss(self, prevBagWord, numLetters = 2, indictBi = -1):
        
        '''
        The idea of incrementally obtain bag-of-patterns comes from 
        
        Patrick Schafer:
        The BOSS is concerned with time series classification in the presence of noise. Data Min. Knowl. Discov. 29(6): 1505-1530 (2015)
        
        This method is suitable to SFA, not to SLA and SAX. We have not used this.
        
        '''
        
        numBitsTrim = bu.numBits(self.discretizer.card) * numLetters
#         if self.binary:
        curBagWord = {}
        for word in prevBagWord.keys():
            if indictBi >= 0:
                isBigram = bu.getBits(word, 0, 1)
                if isBigram:
                    numBitsUni = bu.numBits(self.discretizer.card) * self.discretizer.wordSize
                    newUniWord_0 = bu.trimBits(word, numBitsTrim + numBitsUni + 1)
                    newUniWord_1 = bu.getBits(word, numBitsTrim + 1, numBitsUni - numBitsTrim)
                    newWord = bu.appendBits(newUniWord_0, newUniWord_1, numBitsUni - numBitsTrim)
                    newWord = bu.appendBits(newWord, isBigram, 1)
                    if newWord not in curBagWord.keys():
                        curBagWord[newWord] = set() if self.binary else {}
                        for i in range(2 ** numBitsTrim):
                            affUniword_0 = bu.appendBits(newUniWord_0, i, numBitsTrim)
                            for j in range(2 ** numBitsTrim):
                                affUniword_1 = bu.appendBits(newUniWord_1, j, numBitsTrim)
                                affWord = bu.appendBits(affUniword_0, affUniword_1, numBitsUni)
                                affWord = bu.appendBits(affWord, isBigram, 1)
                                if affWord in prevBagWord.keys():
                                    if self.binary:
                                        curBagWord[newWord] = curBagWord[newWord] | prevBagWord[affWord]
                                    else:
                                        for tsId, cnt in prevBagWord[affWord].items():
                                            if tsId in curBagWord[newWord].keys():
                                                curBagWord[newWord][tsId] += cnt
                                            else:
                                                curBagWord[newWord][tsId] = cnt
                else:
                    newWord_ni = bu.trimBits(word, numBitsTrim + 1)
                    newWord = bu.appendBits(newWord_ni, isBigram, 1)
                    if newWord not in curBagWord.keys():
                        curBagWord[newWord] = set() if self.binary else {}
                        for i in range(2 ** numBitsTrim):
                            affWord = bu.appendBits(newWord_ni, i, numBitsTrim)
                            affWord = bu.appendBits(affWord, isBigram, 1)
                            if affWord in prevBagWord.keys():
                                if self.binary:
                                    curBagWord[newWord] = curBagWord[newWord] | prevBagWord[affWord]
                                else:
                                    for tsId, cnt in prevBagWord[affWord].items():
                                        if tsId in curBagWord[newWord].keys():
                                            curBagWord[newWord][tsId] += cnt
                                        else:
                                            curBagWord[newWord][tsId] = cnt
            else:
                newWord = bu.trimBits(word, numBitsTrim)
                if newWord not in curBagWord.keys():
                    curBagWord[newWord] = set() if self.binary else {}
                    for i in range(2 ** numBitsTrim):
                        affWord = bu.appendBits(newWord, i, numBitsTrim)
                        if affWord in prevBagWord.keys():
                            if self.binary:
                                curBagWord[newWord] = curBagWord[newWord] | prevBagWord[affWord]
                            else:
                                for tsId, cnt in prevBagWord[affWord].items():
                                    if tsId in curBagWord[newWord].keys():
                                        curBagWord[newWord][tsId] += cnt
                                    else:
                                        curBagWord[newWord][tsId] = cnt
        return curBagWord                         
        
    
    def getNewBOP(self, deltaWordSize):
        
        if not deltaWordSize:
            return deepcopy(self)
        
        discretizer = self.discretizer.getNewDiscretizer(deltaWordSize)
        return BOP(discretizer, self.binary)          
    
    def getFeats_Word(self, word, bagTss):
        numTs = len(bagTss)
        feats = np.zeros(numTs)
        for i in range(numTs):
            bagTs = bagTss[i]
            if self.binary and word in bagTs:
                feats[i] = 1
            elif not self.binary and word in bagTs.keys():
                feats[i] = bagTs[word]
        return feats

def getWordFirstBOP_Dataset(bagTss):    #[word: cnt] -> {word: {ts: cnt}}
    
    bagWord = {}
    binary = isinstance(bagTss[0], set)
    for i in range(len(bagTss)):
        bagTs = bagTss[i]
        if binary:
            for word in bagTs:
                if word in bagWord.keys():
                    bagWord[word].add(i)
                else:
                    bagWord[word] = {i}
        else:
            for word, cnt in bagTs.items():
                if word not in bagWord.keys():
                    bagWord[word] = {i: cnt}
                else:
                    bagWord[word][i] = cnt
    return bagWord    