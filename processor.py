import sys

file = open(sys.argv[1]).read().split()

for i,item in enumerate(file):
    print("({}, {})".format(i,item))
