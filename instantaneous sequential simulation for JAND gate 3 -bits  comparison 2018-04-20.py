# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:41:07 2018

@author: zy
"""
import matplotlib.pyplot as plt
import numpy as np
from math import log2

def groupUpdate(I0, A, B,t):
    length = A.shape[0]
    result = [0] * length
    def update(result):
        order = np.arange(length)
        '''np.random.shuffle(order)'''
        for j in range (0, length):
            i = order[j]
            temp = B.item(i,0)
            for k in range (0, length):
                temp = np.add(temp, A.item(i,k) * result[k], casting = "unsafe")
            Itemp = temp * I0
            #print(Itemp)
            result[i] = np.sign(np.random.uniform(-1,1)+np.tanh(Itemp))
            #print(result[i])
        #print(result)
    frequency = [0] * (2**length)
    for t in range (0, t):
        update(result)
        case = 0;
        for i in range (0, length):
            case += (result[i] + 1) / 2 * (2**(length - i - 1))
        frequency[(int)(case)] += 1
    for i in range (0, len(frequency)):
        frequency[i] = frequency[i] / t;
    ''''x = np.arange(len(frequency))
    y = [0]*len(frequency)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, length)
    plt.bar(x, frequency)
    plt.xticks(x+.5, y);
    plt.show()'''
    return frequency, "3*3",t

def groupUpdateClamp0(I0, A, B, index):
    length = A.shape[0]
    result = [0] * length
    def update(result):
        order = np.arange(length)
        '''np.random.shuffle(order)'''
        for j in range (0, length):
            i = order[j]
            temp = B.item(i,0)
            for k in range (0, length):
                temp = np.add(temp, A.item(i,k) * result[k], casting = "unsafe")
            Itemp = temp * I0
            #print(Itemp)
            result[i] = np.sign(np.random.uniform(-1,1)+np.tanh(Itemp))
            result[index] = -1
            #print(result[i])
        #print(result)
    frequency = [0] * (2**length)
    for t in range (0, 100000):
        update(result)
        case = 0;
        for i in range (0, length):
            case += (result[i] + 1) / 2 * (2**(length - i - 1))
        frequency[(int)(case)] += 1
    for i in range (0, len(frequency)):
        frequency[i] = frequency[i] / 100000;
    x = np.arange(len(frequency))
    y = [0]*len(frequency)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, length)
    plt.bar(x, frequency)
    plt.xticks(x+.5, y);
    plt.show()
    return None
    
def groupUpdateClamp1(I0, A, B, index):
    length = A.shape[0]
    result = [0] * length
    def update(result):
        order = np.arange(length)
        '''np.random.shuffle(order)'''
        for j in range (0, length):
            i = order[j]
            temp = B.item(i,0)
            for k in range (0, length):
                temp = np.add(temp, A.item(i,k) * result[k], casting = "unsafe")
            Itemp = temp * I0
            #print(Itemp)
            result[i] = np.sign(np.random.uniform(-1,1)+np.tanh(Itemp))
            result[index] = 1
            #print(result[i])
        #print(result)
    frequency = [0] * (2**length)
    for t in range (0, 100000):
        update(result)
        case = 0;
        for i in range (0, length):
            case += (result[i] + 1) / 2 * (2**(length - i - 1))
        frequency[(int)(case)] += 1
    for i in range (0, len(frequency)):
        frequency[i] = frequency[i] / 100000;
    '''x = np.arange(len(frequency))
    y = [0]*len(frequency)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, length)
    plt.bar(x, frequency)
    plt.xticks(x+.5, y);
    plt.show()'''
    return None

def groupUpdateClamp1Segment(I0, A, B, index1, start,end, t):
    length = A.shape[0]
    result = [0] * length
    def update(result):
        order = np.arange(length)
        '''np.random.shuffle(order)'''
        for j in range (0, length):
            i = order[j]
            temp = B.item(i,0)
            for k in range (0, length):
                temp = np.add(temp, A.item(i,k) * result[k], casting = "unsafe")
            Itemp = temp * I0
            #print(Itemp)
            result[i] = np.sign(np.random.uniform(-1,1)+np.tanh(Itemp))
            result[index1] = 1
            #print(result[i])
        #print(result[start:end+1])
    frequency = [0] * (2**(end-start+1))
    for t in range (0, t):
        update(result)
        case = 0;
        for i in range (start, end+1):
            case += (result[i] + 1) / 2 * (2**(end - i))
        frequency[(int)(case)] += 1
    for i in range (0, len(frequency)):
        frequency[i] = frequency[i] /t;
    
    '''x = np.arange(len(frequency))
    y = [0]*len(frequency)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, end-start+1)
    #plt.bar(x, frequency)'''
    '''fig, ax = plt.subplots()
    rects1 = ax.bar(x, frequency, 0.49, color = 'r')
    rects2 = ax.bar(x + 0.49, frequency, 0.49, color = 'y')
    ax.set_xticks(x+.5)
    ax.set_xticklabels( y);
    ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
    plt.show()'''
    return frequency, "8*8",t
    
def compareFrequency(frequency1, name1, frequency2, name2,t):
    x = np.arange(len(frequency1))
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, frequency1, 0.49, color = 'r')
    rects2 = ax.bar(x + 0.49, frequency2, 0.49, color = 'y')
    y = [0]*len(frequency1)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, (int)(log2(len(frequency1))))
    ax.set_xticks(x+.5)
    ax.set_xticklabels(y);
    ax.legend((rects1[0], rects2[0]), (name1, name2))
    title = "Comparison for " + (str)(t) + " times between " + name1 + " and " + name2
    ax.set_title(title)
    plt.show()


#I = int(input("Enter a number for I:"))
J = np.matrix([[0, -1, 2], [-1, 0, 2], [2, 2, 0]])
J1 = np.matrix([[0,-1, 0, 0, 1, 1, 1, 0],
                [-1, 0, 1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1, -1, 0],
                [0, 1, 0, 0, 1, -1, 1, 0],
                [1, 0, 1, 1, 0, 0, 0, -1],
                [1, 0, 1, -1, 0, 0, 0, 1],
                [1, 0, -1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, -1, 1, 1, 0]])
h1 = np.matrix([[0],[0],[0],[0],[0],[0],[0],[0]])
h = np.matrix([[1], [1], [-2]])
a,b,t = groupUpdate(2,J,h,10000)
c,d,t = groupUpdateClamp1Segment(2, J1, h1,4, 5,7,10000)
compareFrequency(a,b,c,d,t)



