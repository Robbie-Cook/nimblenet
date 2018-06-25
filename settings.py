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

save = True # whether to save the data
inputNodes = 32
hiddenNodes = 16
outputNodes = 32
numInterventions = 10
numPatterns = 20+numInterventions
populationSize = numPatterns-numInterventions
repeats = 50
auto = False
printRate = 5000
numPseudoItems = 128
mymethod = method.catastrophicForgetting

learningConstant = 0.3
momentumConstant = 0.5
errorCriterion = 0.04
maxIterations = 10000

cost_function = sum_squared_error
batch_size = 1
learningAlgorithm = backpropagation_classical_momentum
outputFile = None


# Make a new, distinct file
if save:
    i = 0
    while "output{}.txt".format(i) in os.listdir("data"):
        i+=1
    outputFile = open("data/output{}.txt".format(i), 'w')
