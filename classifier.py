#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 23:39:44 2019

@author: Abu Zahid Bin Aziz
"""

import numpy as np
from keras.models import model_from_json


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
#print("Model loaded")

def seq_data(data):
    # get the maxmium length of the seqence
    max_seq_len = 164
    
    data += "N" *(max_seq_len-len(data))
 
    # tranformation of data set:one_hot encoding
    x_cast = {"A":[[1],[0],[0],[0]],"U":[[0],[1],[0],[0]],\
              "T":[[0],[1],[0],[0]],"G":[[0],[0],[1],[0]],\
              "C":[[0],[0],[0],[1]],"N":[[0],[0],[0],[0]]}
    
    x=[]
    for char in data:
        x.append(x_cast[char])
    
    x= np.array(x)
    x=x.reshape([1,164,4,1])
    #print(x.shape)
    return x

seq= "GTAAGTCTGGGGAGATGGGGGGAGCTCTGCTGAGGGTGCACAAGGCCCTGGCTCTACACACATCCCTGTCTTACAG"
x= seq_data(seq)
p=loaded_model.predict(x)
#print(p[0])
pr=p[0].tolist()
if pr[0]>pr[1]:
    print("It's a Mirtron.")
else:
    print("It's not a Mirtron.")