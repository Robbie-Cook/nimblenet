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

class method:
    random = 0
    recency = 1
    sweep = 2
"""
Parameters:
"""

inputNodes = 4
hiddenNodes = 2
outputNodes = 4
numInterventions = 10
numPatterns = 5+numInterventions
populationSize = numPatterns-numInterventions
repeats = 10
auto = False
learningConstant = 0.3
momentumConstant = 0.5
errorCriterion = 0.04
cost_function = sum_squared_error
batch_size = 1
printRate = 5000
learningAlgorithm = backpropagation_classical_momentum
errorFunction = getGoodness
maxIterations = 10000
mymethod = method.sweep


# def intervene(interveningTask):
#     print("\n\nIntervening\n")
#     interveningInputs = interveningTask['inputPatterns']
#     interveningTeacher = interveningTask['teacher']
#
#     interveningDataset = []
#     for i in range(len(interveningInputs)):
#         interveningDataset.append(Instance(interveningInputs[i], interveningTeacher[i]))
#
#     # Train the network using backpropagation
#     learningAlgorithm(
#             network,                            # the network to train
#             interveningDataset,                      # specify the training set
#             interveningDataset,                          # specify the test set
#             cost_function,                      # specify the cost function to calculate error
#
#             ERROR_LIMIT             = errorCriterion,     # define an acceptable error limit
#             max_iterations         = maxIterations,      # continues until the error limit is reach if this argument is skipped
#
#             batch_size              = batch_size,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
#             print_rate              = printRate,     # print error status every `print_rate` epoch.
#             learning_rate           = learningConstant,      # learning rate
#             momentum_factor         = momentumConstant,      # momentum
#             input_layer_dropout     = 0.0,      # dropout fraction of the input layer
#             hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
#             save_trained_network    = False     # Whether to write the trained weights to disk
#         )

#
# Main routine
#

i = 0
while "output{}.txt".format(i) in os.listdir("data"):
    i+=1
outputFile = open("data/output{}.txt".format(i), 'w')

# Training set
totalError = [0 for i in range(numInterventions+1)]
for i in range(repeats):
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
    print("interventions", interventions)
    print(mytask)
    inputs = mytask.task['inputPatterns']
    teacher = mytask.task['teacher']

    dataset = []
    for i in range(len(inputs)):
        dataset.append(Instance(inputs[i], teacher[i]))

    training_data       = dataset
    test_data           = dataset

    settings            = {
        "n_inputs"              : inputNodes,       # Number of network input signals
        "layers"                : [  (hiddenNodes, sigmoid_function), (outputNodes, sigmoid_function) ],
        "initial_bias_value"    : 0.5,
        "weights_low"           : -0.3,     # Lower bound on the initial weight value
        "weights_high"          : 0.3,
    }


    # initialize the neural network
    network             = NeuralNet( settings )

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

    # print_test(network=network, testset=test_data, cost_function=cost_function)
    # print_test(network=network, testset=test_data, cost_function=cost_function)
    # print("Highest Disparity: ",getHighestDisparity(network=network, testset=test_data, cost_function=cost_function))
    """
    Form inital goodness
    """
    goodness = getGoodness(network=network, testset=test_data, cost_function=cost_function)
    if goodness < 0.9:
        i -= 1
        continue

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
        print("\nRunning Intervention")
        newInstance = Instance(
                                interventions[j]['inputPatterns'][0],
                                interventions[j]['teacher'][0]
                                )
        if mymethod != method.recency:
            random.shuffle(learnt)

        buffer2 = learnt[0]
        buffer3=learnt[0]
        buffer4=learnt[len(learnt)-1]
        if len(alreadyLearned['inputPatterns']) >= 3:
            buffer2=learnt[-1]
            buffer3=learnt[-2]
            buffer4=learnt[-3]

        if mymethod == method.random or mymethod == method.recency:
            interveningDataset = [newInstance, buffer2, buffer3, buffer4]

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
            learnt.append(newInstance)

        if mymethod == method.sweep:
            error = getError(network=network, testset=[newInstance], cost_function=cost_function)
            epochs = 0
            while error > errorCriterion:
                random.shuffle(learnt)
                buffer2 = learnt[0]
                buffer3 = learnt[0]
                buffer4 = learnt[len(learnt)-1]
                if(len(learnt) >= 3):
                    buffer2 = learnt[-1]
                    buffer3 = learnt[-2]
                    buffer4 = learnt[-3]

                interveningDataset = [newInstance, buffer2, buffer3, buffer4]
                # print("dataset", interveningDataset)
                # time.sleep(1)
                # Train the network on one epoch for this buffer
                learningAlgorithm(
                        network,                            # the network to train
                        interveningDataset,                      # specify the training set
                        interveningDataset,                          # specify the test set
                        cost_function,                      # specify the cost function to calculate error

                        ERROR_LIMIT             = 0.000000001,     # define an acceptable error limit
                        max_iterations         = 4,      # continues until the error limit is reach if this argument is skipped

                        batch_size              = 1,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
                        print_rate              = 1000000,     # print error status every `print_rate` epoch.
                        learning_rate           = learningConstant,      # learning rate
                        momentum_factor         = momentumConstant,      # momentum
                        input_layer_dropout     = 0.0,      # dropout fraction of the input layer
                        hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
                        save_trained_network    = False     # Whether to write the trained weights to disk
                    )
                error = getError(network=network, testset=[newInstance], cost_function=cost_function)
                epochs += 1
            print("{}/{} intervention".format(j+1, len(interventions)))
            time.sleep(2)
            learnt.append(newInstance)


        print("Goodness", getGoodness(network=network, testset=test_data, cost_function=cost_function))
        totalError[j+1] += getGoodness(network=network, testset=test_data, cost_function=cost_function)


averageError = [i/repeats for i in totalError]
[outputFile.write(str(i)+"\n") for i in averageError]
