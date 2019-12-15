#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 

import re
import numpy as np

class TSLoader(object):
    
    def loadUCRTimeSeries_2018(self, dataset, sfx, path, padZeros = False):
                    
        #fName = path + '/' + dataset + '/' + dataset + '_preprocessed_' + sfx
        fName = path + '/' + dataset + '/' + dataset + '_' + sfx + '.tsv'

        file = open(fName)
        lines = file.readlines()
        file.close()
        numTs = len(lines)
        tss = []
        labels = []
        tsLens = []
        for line in lines:
            vals = []
            sVals = re.split(r'[,\s]\s*', line.strip())
            for sVal in sVals:
                if sVal != 'NaN':
                    vals.append(float(sVal))
            labels.append(vals[0])
            tss.append(vals[1:])
            tsLens.append(len(vals[1:]))
        labels = np.array(labels, dtype = 'uint32')
        if padZeros:
            maxTsLen = max(tsLens)
            for i in range(numTs):
                tss[i] += [0 for j in range(maxTsLen - len(tss[i]))]
        
        return (tss, labels, numTs, tsLens)