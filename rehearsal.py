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

"""
Takes inputs as Instance objects e.g
recency(Instance([0,0],[0]), Instance([0,1],[0]), Instance([1,0],[0]),
        Instance([1,0],[0]))
"""
def learnBuffer(network, interveningTask, buffer2, buffer3, buffer4, cost_function,
            learningAlgorithm, errorCriterion, maxIterations):
    print("\nIntervening\n")

    interveningDataset = [interveningTask, buffer2, buffer3, buffer4]

    # Train the network using backpropagation
    learningAlgorithm(
            network,                            # the network to train
            interveningDataset,                      # specify the training set
            interveningDataset,                          # specify the test set
            cost_function,                      # specify the cost function to calculate error

            ERROR_LIMIT             = errorCriterion,     # define an acceptable error limit
            max_iterations         = maxIterations,      # continues until the error limit is reach if this argument is skipped

            batch_size              = batch_size,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
            print_rate              = printRate,     # print error status every `print_rate` epoch.
            learning_rate           = learningConstant,      # learning rate
            momentum_factor         = momentumConstant,      # momentum
            input_layer_dropout     = 0.0,      # dropout fraction of the input layer
            hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
            save_trained_network    = False     # Whether to write the trained weights to disk
        )
