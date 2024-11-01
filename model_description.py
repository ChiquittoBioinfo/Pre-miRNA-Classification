#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from keras.models import model_from_json
import tensorflow.keras.backend as K

modelpath = 'original_model/model.json'
weightspath = 'original_model/model.h5'

json_file = open(modelpath, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weightspath)

trainable_count = np.sum([K.count_params(w) for w in loaded_model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in loaded_model.non_trainable_weights])

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
