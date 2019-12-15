#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 

import numpy as np
from TimeSeries.TimeSeriesLoader import TSLoader
from sklearn.metrics import accuracy_score

class Classifier(object):
    
    def __init__(self, inputPath = 'G:/Academic/EclipseWorkspace/HBOP/Data'):
        self.inputPath = inputPath
        self.tsLoader = TSLoader()
            
    def loadUCRDataset_2018(self, dataset, sfx, padZeros = False):
        
        self.dataset = dataset
        if(sfx == 'TRAIN'):
            (self.trainTss, self.trainLabels, self.numTrain, self.trainLens) = self.tsLoader.loadUCRTimeSeries_2018(dataset, sfx, self.inputPath, padZeros)
            self.maxTrainLen = max(self.trainLens)
            self.minTrainLen = min(self.trainLens)
            self.numCls = len(np.unique(self.trainLabels))
            self.setRelabelMap(self.trainLabels)
            self.relabel(self.trainLabels)
            self.numTrainByCls, self.trainIndsByCls = self.getNumByCls(self.trainLabels)
        elif(sfx == 'TEST'):
            (self.testTss, self.testLabels, self.numTest, self.testLens) = self.tsLoader.loadUCRTimeSeries_2018(dataset, sfx, self.inputPath, padZeros)
            self.maxTestLen = max(self.testLens)
            self.minTestLen = min(self.testLens)
            self.relabel(self.testLabels)
            self.numTestByCls, self.testIndsByCls = self.getNumByCls(self.testLabels)
             
    def setRelabelMap(self, labels):
        self.relabelMap = {}
        for label in labels:
            if label not in self.relabelMap.keys():
                    self.relabelMap[label] = len(self.relabelMap)
    
    def getNumByCls(self, labels):
        numByCls = np.zeros(self.numCls, dtype = 'uint32')
        indsByClass = []
        for i in range(self.numCls):
            curInds = np.where(labels == i)[0]
            indsByClass.append(curInds)
            numByCls[i] = len(curInds)
        return numByCls, indsByClass
    
    def relabel(self, labels):
        for i in range(len(labels)):
            labels[i] = self.relabelMap[labels[i]]
    
    def crossValidate(self):
        pass
            
    def train(self, timing = True):
        pass
    
    def test(self, timing = True):
        pass
    
    def getAccuracy(self):
        self.accuracy = accuracy_score(self.testLabels, self.preLabels)
    