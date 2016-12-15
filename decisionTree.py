# Python implementation of a Decision Tree using the Iris dataset (https://en.wikipedia.org/wiki/Iris_flower_data_set)
# Following tutorial from:
# http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html

import numpy as np
from sklearn.datasets import load_iris
from math import log
import random
from math import ceil

class decisionNode:
	def __init__(self, attribute = -1, value = None, results = None, trueBranch = None, falseBranch = None):
		self.attribute = attribute
		self.value = value
		self.results = results
		self.trueBranch = trueBranch
		self.falseBranch = falseBranch

def printTree(iris, tree, indent = ''):
	if tree.results != None:
		print(str(tree.results))
	else:
		print(iris.feature_names[tree.attribute] + ' >= ' + str(tree.value) + '?')
		print(indent + 'T->', end = " ")
		printTree(iris, tree.trueBranch, indent + '  ')
		print(indent + 'F->', end = " ")
		printTree(iris, tree.falseBranch, indent + '  ')

# Join data and their labels in a single matrix
def getIrisData(iris):
	trainingData = []
	for i in range(len(iris.data)):
		trainingData.append(np.append(iris.data[i], iris.target_names[iris.target[i]]))

	trainingData = np.array(trainingData)
	return trainingData

# Split the dataset in training data and test data
def splitData(data):
	np.random.shuffle(data)
	trainingSize = ceil(len(data) * 0.80)
	trainingIndexes = random.sample(range(0, len(data)), trainingSize)
	testIndexes = np.setdiff1d(range(0, len(data)), trainingIndexes)
	
	trainingData = [data[i] for i in trainingIndexes]
	testData = [data[i] for i in testIndexes]

	return (trainingData, testData)

# Divide a set in a set that meets the given condition and another one that doesn't
def divideSet(data, attribute, value):
	splitFunction = lambda data : data[attribute] >= value

	trueSet = [row for row in data if splitFunction(row)]
	falseSet = [row for row in data if not splitFunction(row)]

	return (trueSet, falseSet)

# Counts how many instances of each classa appear in the given dataset
def uniqueCounts(data):
	results = {}
	for row in data:
		result = row[len(row) - 1]
		if result not in results:
			results[result] = 0
		results[result] += 1

	return results

# Calculates the entropy of a given dataset
def entropy(data):
	ent = 0.0
	log2 = lambda x : log(x) / log(2)
	results = uniqueCounts(data)
	for result in results.keys():
		p = float(results[result] / len(data))
		ent -= (p * log2(p))

	return ent

# Build the decision tree
def buildTree(data, depth, maxDepth, scoreFunction = entropy):
	if depth < maxDepth:
		currentScore = scoreFunction(data)
		bestGain = 0.0
		bestCriteria = None
		bestSets = None

		# Substract 1 because last column is the label
		attributeCount = len(data[0]) - 1
		# Divide set for each attribute
		for attr in range(0, attributeCount):
			attributeValues = []
			for row in data:
				if not row[attr] in attributeValues:
					attributeValues.append(row[attr])

			# Divide for each value in the current attribute
			for value in attributeValues:
				(trueSet, falseSet) = divideSet(data, attr, value)

				#Information gain
				# p -> Size of child set relative to parent
				p = float(len(trueSet) / len(data))
				gain = currentScore - (p * scoreFunction(trueSet) + (1 - p) * scoreFunction(falseSet))

				if gain > bestGain and len(trueSet) > 0 and len(falseSet) > 0:
					bestGain = gain
					bestCriteria = (attr, value)
					bestSets = (trueSet, falseSet)

		if bestGain > 0.0:
			trueBranch = buildTree(bestSets[0], depth + 1, maxDepth)
			falseBranch = buildTree(bestSets[1], depth + 1, maxDepth)
			return decisionNode(attribute = bestCriteria[0], value = bestCriteria[1], trueBranch = trueBranch, falseBranch = falseBranch)
		else:
			return decisionNode(results = uniqueCounts(data))
	else:
		return decisionNode(results = uniqueCounts(data))

# Classify
def predict(observation, tree):
	if tree.results != None:
		return tree.results
	else:
		value = observation[tree.attribute]
		if value >= tree.value:
			branch = tree.trueBranch
		else:
			branch = tree.falseBranch

		return predict(observation, branch)

# Get the predicted label
def getPredictedClass(results):
	label = None
	bestLabel = 0
	for key in results.keys():
		if results[key] > bestLabel:
			bestLabel = results[key]
			label = key

	return label

def main():
	MAX_DEPTH = 3
	iris = load_iris()
	data = getIrisData(iris)
	(trainingData, testData) = splitData(data)
	decisionTree = buildTree(trainingData, 0, MAX_DEPTH)
	print('Decision tree:')
	printTree(iris, decisionTree)
	print()

	print('Predicting on test data...')
	correct = 0
	for observation in testData:
		prediction = getPredictedClass(predict(observation, decisionTree))
		actual = observation[len(observation) - 1]
		if prediction == actual:
			correct += 1
		# print('Prediction: ' + prediction + '\tActual: ' + actual)

	accuracy = float(100 * correct / len(testData))
	print()
	print('Accuracy: ' + str(accuracy) + '%')

if __name__ == "__main__":
	main()