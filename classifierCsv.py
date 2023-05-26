#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:28:01 2023

@author: Alisson G. Chiquitto <chiquitto@gmail.com>
"""

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from keras.models import model_from_json
import getopt
import csv

def usage():
    print("USAGE: python classifierCsv.py --csv input.csv --output output.csv")
    print("Example: python classifierCsv.py --csv putative_mirtrons_set.csv --output results.csv")

try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:",["help","csv=","output="])
except getopt.GetoptError:
    print ("Wrong usage!\n")
    usage()
    sys.exit(1)

# parse the options
for op, value in opts:
    if op in ("--csv"):
        input_path = value
    elif op in ("--output"):
        output_path = value
    elif op in ("-h","--help"):
        usage()
        sys.exit()

if len(opts) < 2:
    usage()
    sys.exit(1)

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
    
    # cutting the sequence
    data = data[:max_seq_len]

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

BATCH_SIZE_RESULT = 1000

def flush_results():
    csvfile_writer = open(csvoutput, 'a', newline='')
    csvwriter = csv.DictWriter(csvfile_writer, fieldnames=fieldnames)
    csvwriter.writerows(results)
    csvfile_writer.close()

# Open CSV input
csvinput = open(input_path, newline='')
csvreader = csv.DictReader(csvinput, delimiter=',', quotechar='"')

# Create CSV output
fieldnames = ['id', 'seq', 'ismirtron']
# os.path.splitext(input_path)[0] + '_miRNAClassification.csv'
csvoutput = output_path

csvfile_writer = open(csvoutput, 'w', newline='')
csvwriter = csv.DictWriter(csvfile_writer, fieldnames=fieldnames)
csvwriter.writeheader()
csvfile_writer.close()

results = []
result_counter = 0

for n, row in enumerate(csvreader):

    # seq= "GTAAGTCTGGGGAGATGGGGGGAGCTCTGCTGAGGGTGCACAAGGCCCTGGCTCTACACACATCCCTGTCTTACAG"
    seq= row['seq']
    x= seq_data(seq)

    p=loaded_model.predict(x, verbose=None)
    pr=p[0].tolist()

    # row['ismirtron'] = 1 if pr[0]>pr[1] else 0

    results.append({
        'id': row['id'],
        'seq': row['seq'],
        'ismirtron': 1 if pr[0]>pr[1] else 0
    })
    result_counter += 1

    if result_counter == BATCH_SIZE_RESULT:
        flush_results()
        results.clear()
        result_counter = 0

flush_results()

# Close handlers
csvinput.close()