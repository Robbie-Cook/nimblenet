import task

mytask = (task.Task(
    inputNodes=32,
    hiddenNodes=16,
    outputNodes=32,
    populationSize=20,
    auto=True,
    learningConstant=0.1,
    momentumConstant=0.9
)).task

print("[")
for i in range(0, len(mytask['inputPatterns'])):
    print("[{}, {}], ".format(mytask['inputPatterns'][i],
                                    mytask['teacher'][i]))
print("]")
