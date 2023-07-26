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
    print("USAGE: python classifierCsv.py --csv input.csv --output output.csv --modelpath model")
    print("Example: python classifierCsv.py --csv putative_mirtrons_set.csv --output results.csv --modelpath original_model")

def process_argv():
    requireds = ["csv", "output", "modelpath"]

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
        if op in ("--csv"):
            r['csv'] = value
        elif op in ("--output"):
            r['output'] = value
        elif op in ("--modelpath"):
            r['modelpath'] = value
        elif op in ("-h","--help"):
            usage()
            sys.exit()

    for required in requireds:
        if not required in r:
            print("Wrong usage!!")
            usage()
            sys.exit(1)

    return r

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

def flush_results(fieldnames, results):
    csvfile_writer = open(opts['output'], 'a', newline='')
    csvwriter = csv.DictWriter(csvfile_writer, fieldnames=fieldnames)
    csvwriter.writerows(results)
    csvfile_writer.close()

def main(opts):
    if opts['verbose'] == 1:
        print("input=%s" % opts['csv'])
        print("model=%s" % opts['modelpath'])
        print("output=%s" % opts["output"])

    json_file = open(opts['modelpath'] + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(opts['modelpath'] + "/model.h5")
    #print("Model loaded")

    BATCH_SIZE_RESULT = 1000

    # Open CSV input
    csvinput = open(opts['csv'], newline='')
    csvreader = csv.DictReader(csvinput, delimiter=',', quotechar='"')

    # Create CSV output
    fieldnames = ['id', 'seq', 'ismirtron']

    csvfile_writer = open(opts['output'], 'w', newline='')
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
            flush_results(fieldnames, results)
            results.clear()
            result_counter = 0

    flush_results(fieldnames, results)

    # Close handlers
    csvinput.close()

if __name__ == "__main__":
    # This file contains adaptations for automated execution
    # Example:
    # python classifierCsv.py --csv putative_mirtrons_set.csv --output results.csv --modelpath original_model

    opts = process_argv()
    main(opts)