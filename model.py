#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 23:42:34 2019

@author: Abu Zahid Bin Aziz
"""

import numpy as np
import csv
import random
import math
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten, Concatenate,concatenate
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import  Lambda , Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Conv2D , MaxPooling2D, Convolution2D
from keras.models import model_from_json
import os

def read_data(file1_path,file2_path):
    dataset1_reader = csv.reader(open(file1_path, encoding='utf-8'))
    dataset2_reader = csv.reader(open(file2_path, encoding='utf-8'))
    # define a list to store the data
    all_data_set = []

    # read the data into a list(name,sequence,class)
    for row in dataset1_reader:
        all_data_set.append([row[0],row[1],row[2]])
    for row in dataset2_reader:
        all_data_set.append([row[0],row[1],row[2]])    
    # shuffle the data set randomly    
    random.seed(2)
    random.shuffle(all_data_set)
    return all_data_set

def vectorize_data(dataset):
    # get the maxmium length of the seqence
    max_seq_len = 0
    for item in dataset:
        if len(item[1])>max_seq_len:
            max_seq_len = len(item[1])
    
    # padding with "N" to max_seq_len
    for item in dataset:
        item[1] += "N" *(max_seq_len-len(item[1]))
 
    # tranformation of data set:one_hot encoding
    x_cast = {"A":[[1],[0],[0],[0]],"U":[[0],[1],[0],[0]],\
              "T":[[0],[1],[0],[0]],"G":[[0],[0],[1],[0]],\
              "C":[[0],[0],[0],[1]],"N":[[0],[0],[0],[0]]}
    y_cast = {"TRUE": [1,0],"FALSE":[0,1]} #TRUE:Mirtrons  FALSE:canonical microRN
    
    # define a list to store the vectorized data
    
    x=[]
    y=[]
    for item in dataset:
         data = []
         for char in item[1]:
             data.append(x_cast[char])
         x.append(data)
         y.append(y_cast[item[2]])
    return x,y

def get_model():    
    input1 = keras.layers.Input(shape=(164,4,1))
    c1= Convolution2D(32, (3,4), padding="valid", activation="relu")(input1)
    x1= MaxPooling2D(pool_size=(162,1))(c1)
    c2= Conv2D(32, (4,4), padding="valid", activation="relu")(input1)
    x2= MaxPooling2D(pool_size=(161,1))(c2)
    c3= Conv2D(32, (5,4), padding="valid", activation="relu")(input1)
    x3= MaxPooling2D(pool_size=(160,1))(c3)
    c4= Conv2D(32, (6,4), padding="valid", activation="relu")(input1)
    x4= MaxPooling2D(pool_size=(159,1))(c4)
    c5= Conv2D(32, (7,4), padding="valid", activation="relu")(input1)
    x5= MaxPooling2D(pool_size=(158,1))(c5)
    added = concatenate([x1,x2,x3,x4,x5])
    out=Flatten()(added)
    out=Dense(1024, activation='relu')(out)
    out=keras.layers.Dropout(0.40)(out)
    out=Dense(2, activation='softmax')(out)
    model = keras.models.Model(inputs=[input1], outputs=out)
    x=Adam(lr=0.0001)
    model.compile(optimizer=x, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

FILE_PATH = "miRBase_set.csv"
FILE_PATH_PUTATIVE = "putative_mirtrons_set.csv"
all_data_array = read_data(FILE_PATH,FILE_PATH_PUTATIVE)

x,y = vectorize_data(all_data_array)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.30, random_state=42)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

model= get_model()

model.fit(X_train, y_train, validation_split=0.1, epochs=200,shuffle=False)
scores = model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
