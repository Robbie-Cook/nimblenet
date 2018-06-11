import numpy as np

mylist = [round(a,2) for a in np.arange(0,3.01,0.2)]
print("{" + "{}".format(str(mylist)[1:-1]) + "}")
