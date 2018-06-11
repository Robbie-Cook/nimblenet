from nimblenet.activation_functions import sigmoid_function
from nimblenet.cost_functions import *
from nimblenet.learning_algorithms import *
from nimblenet.neuralnet import NeuralNet
from nimblenet.preprocessing import construct_preprocessor, standarize
from nimblenet.data_structures import Instance
from nimblenet.tools import print_test
import json

mytask = json.loads(open('latestTask.txt').read())
network = NeuralNet.load_network_from_file( "%s.pkl" % "network1" )
prediction_set = [ Instance(item) for item in mytask['inputPatterns'] ]
print(network.predict( prediction_set )) # produce the output signal
