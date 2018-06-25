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
import rehearsal
import settings
import method

"""
Parameters:
"""

save = settings.save
inputNodes = settings.inputNodes
hiddenNodes = settings.hiddenNodes
outputNodes = settings.outputNodes
numInterventions = settings.numInterventions
numPatterns = settings.numPatterns
populationSize = settings.populationSize
repeats = settings.repeats
auto = settings.auto
learningConstant = settings.learningConstant
momentumConstant = settings.momentumConstant
errorCriterion = settings.errorCriterion
cost_function = settings.cost_function
batch_size = settings.batch_size
learningAlgorithm = settings.learningAlgorithm
maxIterations = settings.maxIterations

printRate = settings.printRate
outputFile = settings.outputFile

#
# Main routine
#

# Training set
totalError = [0 for i in range(numInterventions+1)]
for i in range(repeats):
    print("\n{} Repeats completed.\n".format(i))
    mytask = task.Task(
        inputNodes=inputNodes,
        hiddenNodes=hiddenNodes,
        outputNodes=outputNodes,
        populationSize=numPatterns,
        auto=auto,
        learningConstant=learningConstant,
        momentumConstant=momentumConstant
    )

    # Intervening task
    interventions = [mytask.popTask() for a in range(0, numInterventions)]
    inputs = mytask.task['inputPatterns']
    teacher = mytask.task['teacher']

    dataset = []
    for i in range(len(inputs)):
        dataset.append(Instance(inputs[i], teacher[i]))

    training_data       = dataset
    test_data           = dataset

    mysettings            = {
        "n_inputs"              : inputNodes,       # Number of network input signals
        "layers"                : [  (hiddenNodes, sigmoid_function), (outputNodes, sigmoid_function) ],
        "initial_bias_value"    : 0.01,
        "weights_low"           : -0.3,     # Lower bound on the initial weight value
        "weights_high"          : 0.3,
    }


    # initialize the neural network
    network             = NeuralNet( mysettings )

    network.check_gradient( training_data, cost_function )

    # Train the network using backpropagation
    learningAlgorithm(
            network,                            # the network to train
            training_data,                      # specify the training set
            test_data,                          # specify the test set
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

    """
    Form inital goodness
    """
    goodness = getGoodness(network=network, testset=test_data, cost_function=cost_function)
    totalError[0] += goodness

    """
    Run interventions
    """
    alreadyLearned = mytask.task
    learnt = [
        Instance(alreadyLearned['inputPatterns'][i], alreadyLearned['teacher'][i])
        for i in range(len(alreadyLearned['inputPatterns']))
    ]

    for j in range(0, len(interventions)):
        print("\nRunning Intervention", j+1)
        intervention = Instance(
                interventions[j]['inputPatterns'][0],
                interventions[j]['teacher'][0]
                )
        meth = settings.mymethod
        if meth == method.catastrophicForgetting:
            rehearsal.catastrophicForgetting(
                network=network,
                intervention=intervention
                )
        if meth == method.recency:
            rehearsal.recency(
                network=network,
                intervention=intervention,
                learnt=learnt,
                random=False
            )
        elif meth == method.random:
            rehearsal.recency(
                network=network,
                intervention=intervention,
                learnt=learnt,
                random=True
            )
        elif meth == method.pseudo:
            rehearsal.pseudo(
                network=network,
                intervention=intervention,
                numPseudoItems=settings.numPseudoItems
            )
        learnt.append(intervention)


        print("Goodness", getGoodness(network=network, testset=test_data, cost_function=cost_function))
        totalError[j+1] += getGoodness(network=network, testset=test_data, cost_function=cost_function)


averageError = [i/repeats for i in totalError]
if save:
    [outputFile.write(str(i)+"\n") for i in averageError]
