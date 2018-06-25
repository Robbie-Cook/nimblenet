import random
import copy
# import neuralnetwork as nn

class Task:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, populationSize, auto, learningConstant=0.1,
                         momentumConstant=0.9):
        inputList = []
        while len(inputList) < populationSize:
            new_list = []
            for j in range(0, inputNodes):
                new_list.append(random.randint(0,1))
            if new_list not in inputList:
                inputList.append(new_list)

        # if autoassociative, input = output, otherwise output is random
        outputList = []
        if auto:
            outputList = copy.deepcopy(inputList)
        else:
            for i in range(0, populationSize):
                outputList.append([])
                for j in range(0, outputNodes):
                    outputList[i].append(random.randint(0, 1))

        self.task = {}
        self.task["inputPatterns"] = inputList
        self.task["learningConstant"] = learningConstant
        self.task["momentumConstant"] = momentumConstant
        self.task["numberOfHiddenNodes"] = hiddenNodes
        self.task["teacher"] = outputList

    def __str__(self):
        string = ""
        string += "Learning constant: {}\n".format(self.task["learningConstant"])
        string += "Momentum constant: {}\n".format(self.task["momentumConstant"])
        string += ("Input patterns: [") + "\n"
        for row in range(0, len(self.task["inputPatterns"])):
            string += "{}\n".format(self.task["inputPatterns"][row])
        string += "]" + "\n"
        string += ("Teacher patterns: [\n")
        for row in range(0, len(self.task["teacher"])):
            string += ("{}".format(self.task["teacher"][row])) + "\n"
        string += ("]")
        return string


    def createTask(self, inputPatterns, teacher, hiddenNodes,learningConstant, momentumConstant):
        self.task = {}
        self.task["inputPatterns"] = inputPatterns
        self.task["learningConstant"] = learningConstant
        self.task["momentumConstant"] = momentumConstant
        self.task["numberOfHiddenNodes"] = hiddenNodes
        self.task["teacher"] = teacher


    def popTask(self, retain=False):
        newTask = copy.deepcopy(self.task)
        newTask['inputPatterns'] = [newTask['inputPatterns'][0]]
        newTask['teacher'] = [newTask['teacher'][0]]
        if not retain:
            self.task['inputPatterns'] = self.task['inputPatterns'][1:]
            self.task['teacher'] = self.task['teacher'][1:]
        return newTask

    def pushTask(self, t):
        ip = t['inputPatterns']
        tp = t['teacher']
        for i in range(0, len(ip)):
            self.task['inputPatterns'].append(ip[i])
            self.task['teacher'].append(tp[i])

    # def loadTask(self, preserveNet=True):
    #     task = self.task
    #     nn.initalise(input=task["inputPatterns"], teach=task["teacher"], momentum=task["momentumConstant"],
    #              hiddenNodes=task["numberOfHiddenNodes"], learning=task["learningConstant"], preserveNet=preserveNet)
