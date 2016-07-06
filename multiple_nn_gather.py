#!/usr/bin/python
#-*- coding:utf-8 -*-
############################
#File Name: multiple_nn.py
#Author: yuxuan
#Created Time: 2016-05-06 12:21:42
############################
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys
def getMinMax(mat):
    n = len(mat)
    m = 0
    for k in mat[0]:
        m=m+1
    MinNum = [999999999]*m
    MaxNum = [0]*m
    for i in mat:
        for j in range(0,m):
            if i[j] > MaxNum[j]:
                MaxNum[j] = i[j]
    for p in mat:
        for q in range(0,m):
            if p[q] <= MinNum[q]:
                MinNum[q] = p[q]
    return MinNum,MaxNum

def autoNorm(mat,MinNum,MaxNum):
    #MinNum, MaxNum = getMinMax(mat)
    section = list(map(lambda x: x[0]-x[1],zip(MaxNum,MinNum)))
    NormMat=[]

    for kk in mat:
        distance=list(map(lambda x: x[0]-x[1],zip(kk,MinNum)))
        value=list(map(lambda x: x[0]/x[1],zip(distance,section)))
        NormMat.append(value)
    return NormMat

def getData(fileName):
    datafile = open(fileName)
    X_data = []
    Y_data = []
    for i in datafile.readlines():
        line = i.strip().split('\t')
        X_data.append([float(line[0]), float(line[1])])
        Y_data.append([float(line[2])])
    datafile.close()
    return X_data,Y_data

def getSmaller(a,b):
    return (a<=b)
def getBigger(a,b):
    return (a>b)

def getRangeData(fileName,index,value,Fun):
    fr = open(fileName)
    x = []
    y = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #if int(curLine[index]) < int(value):
        if Fun(int(curLine[index]), int(value)):
            x.append([float(curLine[1-index]),float(curLine[index])])#treat as two features
            y.append(float(curLine[-1]))
    fr.close()
    return x,y

def load_data():
        #datafile = open("reduce_datafile")
    if sys.argv[2] == 'total':
        X_data,Y_data = getData(sys.argv[1])
    if sys.argv[2] == 'range':
        X_data,Y_data = getRangeData(sys.argv[1],1,sys.argv[3],getSmaller)
    minNum_x,maxNum_x = getMinMax(X_data)
    X_data = autoNorm(X_data,minNum_x,maxNum_x)
    for i in X_data:
        i.append(1.0)
    
    #minNum_y,maxNum_y = getMinMax(Y_data)
    #data_y = autoNorm(data_y,minNum_y,maxNum_y)

    n_sample = len(X_data)
    sidx = np.random.permutation(n_sample)
    n_train = int(np.round(n_sample * 0.3))
    #test_x_set = np.array([X_data[s] for s in sidx[n_train:]])
    #test_y_set = np.array([Y_data[s] for s in sidx[n_train:]])
    test_x_set = np.array([X_data[s] for s in sidx[:n_train]])
    test_y_set = np.array([Y_data[s] for s in sidx[:n_train]])
    train_x_set = np.array(X_data)
    train_y_set = np.array(Y_data)
    #train_x_set = np.array([X_data[s] for s in sidx])
    #train_y_set = np.array([Y_data[s] for s in sidx])
    '''
    ax = plt.subplot(111,projection='3d')
    ax.scatter(train_x_set[:,0],train_x_set[:,1],train_y_set.T[0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    '''
    #return train_x_set[:,1:3], train_y_set, test_x_set[:,1:3], test_y_set
    return train_x_set, train_y_set, test_x_set, test_y_set



model = Sequential()

model.add(Dense(10, input_dim=3, W_regularizer=l2(0.005)))
model.add(Activation('linear'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
#model.add(Dense(10, input_dim = 6, W_regularizer=l2(0.005)))
#model.add(Activation('linear'))

model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))

model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))

model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))
model.add(Dense(10, input_dim = 10, W_regularizer=l2(0.005)))
model.add(Activation('relu'))



model.add(Dense(1, input_dim = 6, W_regularizer=l2(0.005)))
model.add(Activation('linear'))

model.compile(loss="mean_squared_error", optimizer="rmsprop")
train_x_set, train_y_set, test_x_set, test_y_set = load_data()

model.fit(train_x_set, train_y_set, batch_size=1000, nb_epoch=10000, validation_split=0.05)
print(model.get_weights())
predicted = model.predict(test_x_set)
rmse = np.sqrt(((predicted - test_y_set) ** 2).mean())


print("预测值:")
print(predicted.T)
print("实际:")
print(test_y_set.T)
print(rmse)

#num = 0



#for i in range(len(test_y_set)):
#    if np.sqrt((predicted[i]-test_y_set[i])**2)<0.1*test_y_set[i]:
#        num+=1
#        print str(predicted[i])+" "+str(test_y_set[i])+"\n"
#print "num:"+str(num)
#print "corre"+str(np.corrcoef(predicted,test_y_set,rowvar=0)[0,1])
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
ax = plt.subplot()
ax.set_xlabel("true value")
ax.set_ylabel("predicted value")
ax.set_title(sys.argv[3])
ax.scatter(test_y_set,predicted)
#x = np.linspace()
ax.plot(test_y_set,test_y_set,'r')
plt.show()
'''
ax = plt.subplot(211,projection='3d')
ax.scatter(test_x_set[:,0],test_x_set[:,1],test_y_set)
ax_1 = plt.subplot(212,projection='3d')
ax_1.scatter(test_x_set[:,0],test_x_set[:,1],predicted)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
'''
'''
print len(test_x_set[:,0])
print len(test_y_set)
print len(predicted)
ax = plt.subplot(211)
ax.scatter(test_x_set[:,0],test_y_set)
ax_1 = plt.subplot(212)
ax_1.scatter(test_x_set[:,0],predicted)
plt.show()
'''
