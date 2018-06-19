import numpy as np

class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, target = None ):
        self.features = np.array(features)

        if target != None:
            self.targets  = np.array(target)
        else:
            self.targets  = None

    def __str__(self):
        if len(self.targets) == 0:
            return "({})".format(self.features)
        else:
            return "({}, {})".format(self.features, self.targets)
#endclass Instance
