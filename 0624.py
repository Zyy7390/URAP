# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:41:07 2018

@author: zy
"""
import matplotlib.pyplot as plt
import numpy as np
from math import log2
import xlrd
import xlwt
import numpy as np


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
            result[index1] = -1
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
    
    x = np.arange(len(frequency))
    y = [0]*len(frequency)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, end - start + 1 )
    plt.bar(x, frequency)
    plt.xticks(x+.5, y);
    plt.show()
    return frequency, "8*8",t
    
def compareFrequency(frequency1, name1, frequency2, name2,t):
    x = np.arange(len(frequency1))
    fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = [24,16]
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
JAND = np.matrix([[0,-1, 0, 0, 1, 1, 1, 0],
                [-1, 0, 1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 1, -1, 0],
                [0, 1, 0, 0, 1, -1, 1, 0],
                [1, 0, 1, 1, 0, 0, 0, -1],
                [1, 0, 1, -1, 0, 0, 0, 1],
                [1, 0, -1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, -1, 1, 1, 0]])
JXOR = np.matrix([[0,0,0,0,1,-1,1,-1],
                [0,0,0,0,-1,-1,-1,-1],
                [0,0,0,0,1,-1,-1,1],
                [0,0,0,0,1,1,-1,-1],
                [1,-1,1,1,0,0,0,0],
                [-1,-1,-1,1,0,0,0,0],
                [1,-1,-1,-1,0,0,0,0],
                [-1,-1,1,-1,0,0,0,0]])
J3 = np.matrix([[1,-1,1,-1],
      [-1,-1,-1,-1],
      [1,-1,-1,1],
      [1,1,-1,-1]])
h2 = np.matrix([[0],[0],[0],[0]])
h1 = np.matrix([[0],[0],[0],[0],[0],[0],[0],[0]])
h = np.matrix([[1], [1], [-2]])

filename = "aa1.xlsx"
file = xlrd.open_workbook(filename)
content = file.sheet_by_index(0)

FA33 = [[0]*33 for i in range(0,33)]
for i in range(0,33):
    for j in range(0,33):
        FA33[i][j] = content.cell_value(i+1,j+1)
        
h33 = [[0]]*33

FA33 = np.matrix(FA33)
h33 = np.matrix(h33)
'''print(FA33)'''

'''P = True
for i in range(0,33):
    for j in range(0,33):
        if (FA33.item(i,j) != FA33.item(j,i)):
            P = False
            break
        break
print(P)'''

def groupUpdate1SegmentFA33(I0, A, B, start,end, t):
    length = A.shape[0]
    result = [0] * length
    def update(result):
        '''order = np.arange(length)
        ''''''
        for j in range (0, length):
            i = order[j]
            temp = B.item(i,0)
            for k in range (0, length):
                temp = np.add(temp, A.item(i,k) * result[k], casting = "unsafe")
            Itemp = temp * I0
            #print(Itemp)'''
        current = A.dot(result)
        for i in range(0, length):
            result[i] = np.sign(np.random.uniform(-1,1)+np.tanh(current.item(i)))
            result[4] = 1
            result[10] = 1
            result[15] = 1
            result[21] = 1  
            result[27] = -1
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
    
    x = np.arange(len(frequency))
    y = [0]*len(frequency)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, end - start + 1 )
    plt.bar(x, frequency)
    plt.xticks(x+0.5, y);
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [24,16]
    plt.show()
    
    return frequency, "33*33",t
    
#groupUpdate1SegmentFA33(2, FA33, h33, 28,32, 1000000)
def generateMultiFAMatrix(FAMATRIX) {
    '''This should serve as a function to utilize one single FA matrix to generate correspondoing matrix for stringing multiple FA gate together'''
    
}
JFA = [[0,0,0,0,0,0,0,4,-1,-1,-1,-1,-2,-1],
       [0,0,0,0,0,0,4,0,-1,-1,2,-1,1,-1],
       [0,0,0,0,0,4,0,0,-1,-1,-1,2,1,-1],
       [0,0,0,0,4,0,0,0,-1,-2,1,1,-1,1],
       [0,0,0,4,0,0,0,0,-1,2,-1,-1,1,-1],
       [0,0,4,0,0,0,0,0,-1,1,1,-2,-1,1],
       [0,4,0,0,0,0,0,0,-1,1,-2,1,-1,1],
       [4,0,0,0,0,0,0,0,-1,1,1,1,2,1],
       [-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0],
       [-1,-1,-1,-2,2,1,1,1,0,0,-1,-1,1,2],
       [-1,2,-1,1,-1,1,-2,1,0,-1,0,-1,1,2],
       [-1,-1,2,1,-1,-2,1,1,0,-1,-1,0,1,2],
       [-2,1,1,-1,1,-1,-1,2,0,1,1,1,0,-2],
       [-1,-1,-1,1,-1,1,1,1,0,2,2,2,-2,0]]
h14 = [[0]]*14
FA14 = np.matrix(JFA)
h14 = np.matrix(h14)

def groupUpdate1SegmentFA14(I0, A, B, start,end, t):
    length = A.shape[0]
    result = [0] * length
    def update(result):
        '''order = np.arange(length)
        '''
        '''
        for j in range (0, length):
            i = order[j]
            temp = B.item(i,0)
            for k in range (0, length):
                temp = np.add(temp, A.item(i,k) * result[k], casting = "unsafe")
            Itemp = temp * I0
            #print(Itemp)
            result[i] = np.sign(np.random.uniform(-1,1)+np.tanh(Itemp))
            result[8] = 1'''
        current = A.dot(result)
        for i in range(0, length):
            result[i] = np.sign(np.random.uniform(-1,1)+np.tanh(current.item(i)))
            result[8] = -1
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
    
    x = np.arange(len(frequency))
    y = [0]*len(frequency)
    for i in range(0,len(y)):
        y[i] = "{:0{}b}".format(i, end - start + 1 )
    plt.bar(x, frequency)
    plt.rcParams["figure.figsize"] = [24,16]
    plt.xticks(x+.5, y);
    plt.savefig("0430.png");
    plt.show()
    return frequency, "14*14",t
    
#groupUpdate1SegmentFA14(2, FA14, h14, 9,13, 1000000)
import time
start_time = time.time()
a,b,t = groupUpdate1SegmentFA14(2, FA14, h14, 9,13, 1000000)
#c,d,t = groupUpdate1SegmentFA33(2, FA33, h33, 28,32, 1000000)
#compareFrequency(a,b,c,d,t)
print((str)(time.time() - start_time) + "s")



