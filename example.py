from nimblenet.activation_functions import sigmoid_function
from nimblenet.cost_functions import *
from nimblenet.learning_algorithms import *
from nimblenet.neuralnet import NeuralNet
from nimblenet.preprocessing import construct_preprocessor, standarize
from nimblenet.data_structures import Instance
from nimblenet.tools import print_test
import task
import os
import json


# Training set

taskobject = task.Task(
    inputNodes=8,
    hiddenNodes=4,
    outputNodes=8,

    populationSize=20,
    auto=False,
    momentumConstant=0.5,
    learningConstant=0.3)

mytask = taskobject.task

i0 = taskobject.popTask()

dataset = []
print("Task" + str(taskobject))
for i in range(len(mytask['inputPatterns'])):
    dataset.append(Instance(mytask['inputPatterns'][i], mytask['teacher'][i]))

preprocess          = construct_preprocessor( dataset, [standarize] )
training_data       = preprocess( dataset )
test_data           = preprocess( dataset )

cost_function       = cross_entropy_cost
settings            = {
    # Required settings
    "n_inputs"              : len(mytask['inputPatterns'][0]),       # Number of network input signals
    "layers"                : [  (mytask['numberOfHiddenNodes'], sigmoid_function), (len(mytask['teacher'][0]), sigmoid_function) ],
                                        # [ (number_of_neurons, activation_function) ]
                                        # The last pair in the list dictate the number of output signals

    # Optional settings
    "initial_bias_value"    : 0.0,
    "weights_low"           : -0.1,     # Lower bound on the initial weight value
    "weights_high"          : 0.1,      # Upper bound on the initial weight value
}


# initialize the neural network
network             = NeuralNet( settings )

network.check_gradient( training_data, cost_function )

# Train the network using backpropagation
RMSprop(
        network,                            # the network to train
        training_data,                      # specify the training set
        test_data,                          # specify the test set
        cost_function,                      # specify the cost function to calculate error

        ERROR_LIMIT             = 0.02,     # define an acceptable error limit
        #max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped

        batch_size              = 0,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
        print_rate              = 1000,     # print error status every `print_rate` epoch.
        learning_rate           = mytask['learningConstant'],      # learning rate
        momentum_factor         = mytask['momentumConstant'],      # momentum
        input_layer_dropout     = 0.0,      # dropout fraction of the input layer
        hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
        save_trained_network    = False     # Whether to write the trained weights to disk
    )
# Print a network test
print_test( network, training_data, cost_function, errorOnly=True )


dataset2 = [Instance(i0['inputPatterns'][0], i0['teacher'][0])]

preprocess2          = construct_preprocessor( dataset2, [standarize] )
training_data2       = preprocess( dataset2 )
test_data2           = preprocess( dataset2 )

# RMSprop(
#         network,                            # the network to train
#         training_data2,                      # specify the training set
#         test_data2,                          # specify the test set
#         cost_function,                      # specify the cost function to calculate error
#
#         ERROR_LIMIT             = 0.02,     # define an acceptable error limit
#         #max_iterations         = 100,      # continues until the error limit is reach if this argument is skipped
#
#         batch_size              = 0,        # 1 := no batch learning, 0 := entire trainingset as a batch, anything else := batch size
#         print_rate              = 1000,     # print error status every `print_rate` epoch.
#         learning_rate           = 0.3,      # learning rate
#         momentum_factor         = 0.5,      # momentum
#         input_layer_dropout     = 0.0,      # dropout fraction of the input layer
#         hidden_layer_dropout    = 0.0,      # dropout fraction in all hidden layers
#         save_trained_network    = False     # Whether to write the trained weights to disk
#     )


# Prediction Example
# """
# prediction_set = [ Instance(item) for item in mytask['inputPatterns'] ]
# prediction_set = preprocess( prediction_set )
#
# print('\n')
# print network.predict( prediction_set ) # produce the output signal
