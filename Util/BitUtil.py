#coding=gbk
#coding=utf-8
#-*- coding: UTF-8 -*- 
import numpy as np

def numBits(intVal):
    return int(np.ceil(np.log2(intVal)))

def appendBits(intVal, newBits, shift):
    return int(int(int(intVal) << int(shift)) | int(newBits))

def trimBits(intVal, shift):
    return int(int(intVal) >> int(shift))

def setBits(intVal, newBits, shift, width = None, isSetTo0 = False):
    if not isSetTo0:
        if width is None:
            width = numBits(newBits)
        intVal = setBitsTo0(intVal, shift, width)
    return int(int(intVal) | (int(newBits) << int(shift)))

def setBitsTo1(intVal, shift, numBits): 
    return setBits(intVal, 2 ** numBits - 1, shift)
    
def setBitsTo0(intVal, shift, numBits):  
    return int(int(intVal) & ~(2 ** int(numBits) - 1 << int(shift)))
    
def getBits(intVal, shift, numBits):
    return int((int(intVal) & (2 ** int(numBits) - 1 << int(shift))) >> int(shift))
