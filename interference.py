def intervene(interveningTask):
    print("\n\n\nIntervening\n\n\n")
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

def recency():
