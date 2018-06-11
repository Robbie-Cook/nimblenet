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




"""
Parameters:
"""



inputNodes = 32
hiddenNodes = 16
outputNodes = 32
numInterventions = 10
populationSize = 20+numInterventions
auto = True
learningConstant = 0.3
momentumConstant = 0.5
errorCriterion = 0.01
cost_function = sum_squared_error
batch_size = 1
printRate = 5000
learningAlgorithm = backpropagation_classical_momentum
repeats = 50


def intervene(interveningTask):
    print("\n\nIntervening\n")
    interveningInputs = interveningTask['inputPatterns']
    interveningTeacher = interveningTask['teacher']

    interveningDataset = []
    for i in range(len(interveningInputs)):
        interveningDataset.append(Instance(interveningInputs[i], interveningTeacher[i]))

    # Train the network using backpropagation
    learningAlgorithm(
            network,                            # the network to train
            interveningDataset,                      # specify the training set
            interveningDataset,                          # specify the test set
            cost_function,                      # specify the cost function to calculate error

            ERROR_LIMIT             = errorCriterion,     # define an acceptable error limit
            max_iterations         = 10000,      # continues until the error limit is reach if this argument is skipped

            batch_size              = batch_size,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
            print_rate              = printRate,     # print error status every `print_rate` epoch.
            learning_rate           = learningConstant,      # learning rate
            momentum_factor         = momentumConstant,      # momentum
            input_layer_dropout     = 0.0,      # dropout fraction of the input layer
            hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
            save_trained_network    = False     # Whether to write the trained weights to disk
        )

# def recency(newTask):
#     print("\n\nRecency\n")
#     buffer = [Instance(newTask['inputPatterns'][0],
#                 dataset[-3],
#                 dataset[-2],
#                 dataset[-1]]
#     # Train the network using backpropagation
#     learningAlgorithm(
#             network,                            # the network to train
#             interveningDataset,                      # specify the training set
#             interveningDataset,                          # specify the test set
#             cost_function,                      # specify the cost function to calculate error
#
#             ERROR_LIMIT             = errorCriterion,     # define an acceptable error limit
#             max_iterations         = 10000,      # continues until the error limit is reach if this argument is skipped
#
#             batch_size              = batch_size,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
#             print_rate              = printRate,     # print error status every `print_rate` epoch.
#             learning_rate           = learningConstant,      # learning rate
#             momentum_factor         = momentumConstant,      # momentum
#             input_layer_dropout     = 0.0,      # dropout fraction of the input layer
#             hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
#             save_trained_network    = False     # Whether to write the trained weights to disk
#         )

i = 0
while "output{}.txt".format(i) in os.listdir("."):
    i+=1
outputFile = open("output{}.txt".format(i), 'w')

# Training set

for i in range(repeats):
    mytask = task.Task(
        inputNodes=inputNodes,
        hiddenNodes=hiddenNodes,
        outputNodes=outputNodes,
        populationSize=populationSize,
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

    # for i in range(len(mytask['inputPatterns'])):
    #     dataset.append(Instance(mytask['inputPatterns'][i], mytask['teacher'][i]))

    preprocess          = construct_preprocessor( dataset, [standarize] )
    training_data       = dataset#preprocess( dataset )
    test_data           = dataset#preprocess( dataset )

    settings            = {
        # Required settings
        "n_inputs"              : inputNodes,       # Number of network input signals
        "layers"                : [  (hiddenNodes, sigmoid_function), (outputNodes, sigmoid_function) ],
                                            # [ (number_of_neurons, activation_function) ]
                                            # The last pair in the list dictate the number of output signals
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
            max_iterations         = 100000,      # continues until the error limit is reach if this argument is skipped

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
    print("Highest Disparity: ",getHighestDisparity(network=network, testset=test_data, cost_function=cost_function))
    outputFile.write(str(getError(network=network, testset=test_data, cost_function=cost_function))+"\n")

    """
    Intervening trial
    """

    for i in range(0, len(interventions)):
        intervene(interventions[i])
        outputFile.write(str(getError(network=network, testset=test_data, cost_function=cost_function))+"\n")
        print(getError(network=network, testset=test_data, cost_function=cost_function))
    outputFile.write("\n")
    outputFile.flush()
