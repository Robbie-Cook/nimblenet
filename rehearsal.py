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
import random as rand
import settings

"""
Method to train a buffer of items e.g. [Instance([0,1], [1,0]), Instance([1,0], [1,0])]
to a given criterion.
"""
def trainBuffer(network, buffer, maxIterations=settings.maxIterations, quiet=False):
    # Train the network using backpropagation
    settings.learningAlgorithm(
            network,                            # the network to train
            buffer,                      # specify the training set
            buffer,                          # specify the test set
            settings.cost_function,                      # specify the cost function to calculate error
            quiet=quiet,
            ERROR_LIMIT             = settings.errorCriterion,     # define an acceptable error limit
            max_iterations         = maxIterations,      # continues until the error limit is reach if this argument is skipped

            batch_size              = settings.batch_size,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
            print_rate              = settings.printRate,     # print error status every `print_rate` epoch.
            learning_rate           = settings.learningConstant,      # learning rate
            momentum_factor         = settings.momentumConstant,      # momentum
            input_layer_dropout     = 0.0,      # dropout fraction of the input layer
            hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
            save_trained_network    = False     # Whether to write the trained weights to disk
        )


"""
Catastrophic Forgetting
"""
def catastrophicForgetting(network, intervention):
    newInstance = intervention
    interveningDataset = [newInstance]
    trainBuffer(network, interveningDataset)

"""
Recency rehearsal (or random if random=True)
"""
def recency(network, intervention, learnt, random=False):
    newInstance = intervention

    if random:          # Random rehearsal
        rand.shuffle(learnt)

    buffer2=learnt[0]
    buffer3=learnt[0]
    buffer4=learnt[len(learnt)-1]
    if len(learnt) >= 3:
        buffer2=learnt[-1]
        buffer3=learnt[-2]
        buffer4=learnt[-3]


    interveningDataset = [newInstance, buffer2, buffer3, buffer4]
    trainBuffer(network, interveningDataset)

"""
Random rehearsal
"""
def random(network, intervention, learnt):
    recency(network, intervention, learnt, random=True)

"""
Random pseudorehearsal
"""
def pseudo(network, intervention, numPseudoItems):
    # Pseudoitems for pseudorehearsal
    pseudoItems = generatePseudoPairs(numPseudoItems)
    rand.shuffle(pseudoItems)
    buffer = [intervention, pseudoItems[-1], pseudoItems[-2], pseudoItems[-3]]
    trainBuffer(network, buffer)

"""
Generates pseudoitem pairs (input and output) for pseudorehearsal
"""
def generatePseudoPairs(numItems):
    mytask = task.Task(inputNodes=settings.inputNodes, hiddenNodes=settings.hiddenNodes,
                outputNodes=settings.outputNodes, populationSize=numItems, auto=False).task
    pseudoInputs = mytask['inputPatterns']
    pseudoItems = [Instance(a, getOutputs(network, a)) for a in pseudoInputs]
    return pseudoItems

"""
Sweep rehearsal
"""
def sweep(network, intervention, learnt):
    iterations = 0
    currentError = getError(network, [intervention], settings.cost_function)
    while currentError > settings.errorCriterion and iterations < settings.maxIterations:
        iterations+=1
        if iterations % 5000 == 0:
            print("{} times, error: {}, goodness: {}".format(iterations,  getError(network, [intervention], settings.cost_function),
            getGoodness(network, [intervention])))
        rand.shuffle(learnt)
        if len(learnt) >= 3:
            buffer = [intervention, learnt[-1], learnt[-2], learnt[-3]]
        else:
            buffer = [intervention, intervention, intervention, intervention]
        rand.shuffle(buffer)
        trainBuffer(network, buffer, maxIterations=1, quiet=True)

        # update current error
        currentError = getError(network, [intervention], settings.cost_function)

def pseudoSweep(network, intervention, numPseudoItems):
    iterations = 0
    currentError = getError(network, [intervention], settings.cost_function)
    while currentError > settings.errorCriterion and iterations < settings.maxIterations:
        iterations+=1
        if iterations % settings.printRate == 0:
            print("{} times, error: {}, goodness: {}".format(iterations,  getError(network, [intervention], settings.cost_function),
            getGoodness(network, [intervention])))
        rand.shuffle(learnt)
        pseudoItems = generatePseudoPairs(numPseudoItems)
        rand.shuffle(pseudoItems)
        if len(learnt) >= 3:
            buffer = [intervention, pseudoItems[-1], pseudoItems[-2], pseudoItems[-3]]
        else:
            buffer = [intervention, intervention, intervention, intervention]
        rand.shuffle(buffer)
        trainBuffer(network, buffer, maxIterations=1, quiet=True)

        # update current error
        currentError = getError(network, [intervention], settings.cost_function)
