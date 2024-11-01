#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 23:42:34 2019

@author: Abu Zahid Bin Aziz
"""

import numpy as np
import csv
import random
import keras
from keras.layers import Dense, Flatten, concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D , MaxPooling2D, Convolution2D
from keras.models import model_from_json
import os, sys
import getopt

def read_data(file1_path,file2_path):
    dataset1_reader = csv.DictReader(open(file1_path, encoding='utf-8'),
                                     delimiter=',', quotechar='"')
    dataset2_reader = csv.DictReader(open(file2_path, encoding='utf-8'),
                                     delimiter=',', quotechar='"')

    # define a list to store the data
    all_data_set = []

    # read the data into a list(name,sequence,class)
    for row in dataset1_reader:
        all_data_set.append([row['id'],row['seq'],row['ismirtron']])
    for row in dataset2_reader:
        all_data_set.append([row['id'],row['seq'],row['ismirtron']])    
    
    # shuffle the data set randomly    
    random.seed(2)
    random.shuffle(all_data_set)
    
    return all_data_set

def vectorize_data(dataset):
    # # get the maxmium length of the seqence
    # max_seq_len = 0
    # for item in dataset:
    #     if len(item[1])>max_seq_len:
    #         max_seq_len = len(item[1])

    # set the maxmium length of the seqence
    # for cnn input
    max_seq_len = 164
    
    # cutting and padding with "N" to max_seq_len
    for item in dataset:
        item[1] = item[1][:max_seq_len]
        item[1] += "N" * (max_seq_len-len(item[1]))
 
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

def get_model_json():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    x=Adam(lr=0.0001)
    loaded_model.compile(optimizer=x, loss='categorical_crossentropy', metrics=['accuracy'])

    return loaded_model

def usage():
    print("USAGE: python model_chiquitto.py --neg miRBase_set.csv --pos putative_mirtrons_set.csv --modelpath model")

def process_argv():
    requireds = ["neg", "pos", "modelpath"]

    try:
        longopts = [ opt + "=" for opt in requireds ]
        opts, args = getopt.getopt(sys.argv[1:], "", longopts)
    except getopt.GetoptError:
        print("Wrong usage!")
        usage()
        sys.exit(1)

    # parse the options
    r = { 'verbose': 1 }
    for op, value in opts:
        if op in ("--neg"):
            r['neg'] = value
        elif op in ("--pos"):
            r['pos'] = value
        elif op in ("--modelpath"):
            r['modelpath'] = value
        elif op in ("--verbose"):
            r['verbose'] = int(value)
        elif op in ("-h","--help"):
            usage()
            sys.exit()

    for required in requireds:
        if not required in r:
            print("Wrong usage!!")
            usage()
            sys.exit(1)

    return r

def save_model(model, modelpath):
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    model_json = model.to_json()
    with open(modelpath + "/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(modelpath + "/model.h5")

def main(opts):
    # fix random seed for reproducibility
    # numpy.random.seed(42)

    FILE_PATH = opts["neg"] # "miRBase_set.csv"
    FILE_PATH_PUTATIVE = opts["pos"] # "putative_mirtrons_set.csv"

    if opts['verbose'] == 1:
        print(f"neg={FILE_PATH}")
        print(f"pos={FILE_PATH_PUTATIVE}")
        print("output=%s" % opts["modelpath"])

    all_data_array = read_data(FILE_PATH,FILE_PATH_PUTATIVE)

    x,y = vectorize_data(all_data_array)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.0, random_state=42)
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    X_train = np.array(x)
    y_train = np.array(y)

    model= get_model()
    # model= get_model_json()

    # model.fit(X_train, y_train, validation_split=0.1, epochs=200, shuffle=False)
    model.fit(X_train, y_train, epochs=200, shuffle=False, verbose=2)

    save_model(model, opts["modelpath"])

    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

if __name__ == "__main__":
    # This file contains adaptations for automated execution
    # Example:
    # python model_chiquitto.py --neg miRBase_set.csv --pos putative_mirtrons_set.csv --modelpath chiquitto

    opts = process_argv()
    main(opts)