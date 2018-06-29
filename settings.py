"""
Global settings
"""

from nimblenet.activation_functions import sigmoid_function
from nimblenet.cost_functions import *
from nimblenet.learning_algorithms import *
from nimblenet.neuralnet import NeuralNet
from nimblenet.preprocessing import construct_preprocessor, standarize
from nimblenet.data_structures import Instance
from nimblenet.tools import *
import task
import os
import json
import numpy as np
import random
import time
import settings
import method
import getopt
import sys

"""
Settings.py -- a file with all of the project settings.
This includes the rehearsal algorithm used, number of neurons, etc.

These settings are implemented in `main.py` and `rehearsal.py`
"""

save = True # whether to save the data

"""
The rehearsal algorithm to use.
Options are:
    - method.catastrophicForgetting (No rehearsal)
    - method.random (Random rehearsal)
    - method.recency (Recency rehearsal)
    - method.sweep (Sweep rehearsal)
    - method.pseudo (Random pseudorehearsal)
    - method.pseudoSweep (Sweep pseudorehearsal)
"""
mymethod = method.random

numPseudoItems = 128 # How many pseudoitems to generate

inputNodes = 32          # Number of input neurons
hiddenNodes = 16         # Number of hidden neurons
outputNodes = 32         # Number of output neurons
numInterventions = 10    # Number of intervening trials
numPatterns = 20+numInterventions  # Total number of patterns to learn

populationSize = numPatterns-numInterventions
repeats = 50 # Number of times to repeat the experiment completely on a new population
             # including new intervening trials and a new network

auto = False     # Whether the learning is autoassociative (e.g. [1,0] -> [1,0])
                 # or heteroassociative (e.g. [1,0] -> [0,0])
                 # Input patterns are uniquely generated

printRate = 5000

learningConstant = 0.3
momentumConstant = 0.5
errorCriterion = 0.04
maxIterations = 10000

cost_function = sum_squared_error
batch_size = 1
learningAlgorithm = backpropagation_classical_momentum
outputFile = None

if not __name__ == "__main__":
    # Make a new, distinct file
    if save:
        i = 0
        while "output{}.txt".format(i) in os.listdir("data"):
            i+=1
        outputFile = open("data/output{}.txt".format(i), 'w')
