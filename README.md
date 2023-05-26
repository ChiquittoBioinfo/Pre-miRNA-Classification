# Pre-miRNA-Classification

Dependencies:

1. Python
2. Numpy
3. Sci-kit learn
4. tensorflow
5. Keras

Description:

models.py -> contains the structure of the model.

classifier.py -> works as a predictor for nucleotide sequence input.

surface.py -> surface plot of the grid search results.

## Install

You need to create a new conda environment.

```bash
conda create -n miRNAClassification python=3.10.11 -y
conda install -n miRNAClassification -c anaconda -y numpy
conda install -n miRNAClassification -c conda-forge -y keras
conda install -n miRNAClassification -c anaconda -y tensorflow
```

Install `keras-gpu` if you have a GPU.

```bash
conda install -n miRNAClassification -c anaconda keras-gpu
conda install -n miRNAClassification -c anaconda -y tensorflow-gpu
```

## Running

First, activate the environment.

```bash
conda activate miRNAClassification
```

To run the original script:

```bash
python classifier.py
```

To run the CSV classifier version

```bash
python classifierCsv.py --csv data/generated01.csv --output data/results01.csv
```

## Removing the conda env

To remove the miRNAClassification environment, run:

```bash
conda remove --name miRNAClassification --all
```
