import os
import sys
import numpy as np
import scipy as sp
import scipy.io as spio
import sklearn
from sklearn import svm
import pandas as pd
import glob
import EEGFunctions
import csv
import dataIO

training_data = {}

# Train a classifier.
def DoTraining(p_id, p_method):




if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Train an sklearn algorithm and/or apply it to make predictions')
	parser.add_argument('id', type=str, help='Unique id for saving/loading the model from pickle file')
	#parser.add_argument('--preprocess', type=str, help=': separated list of features to process')
	parser.add_argument('--train', type=str, help='Name of sklearn algorithm')
	parser.add_argument('--test', action='store_true', help='Load model from id and apply to test data')
	parser.add_argument('--subject', type=str, help='Run over a single subject only')

	# Specify subjects
	subjects = []
	if args.subject:
		subjects.append(args.subject)
	else:
		subjects = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8',]

	# Load data
	for subject in subjects:
		training_data[subject] = {}
		if args.train:
			training_data[subject["interictal"] = dataIO.LoadPanelFromPickle(subject, "interictal", True)
			training_data[subject["ictal"]      = dataIO.LoadPanelFromPickle(subject, "ictal", True)
		if args.test:
			training_data[subject["test"] = dataIO.LoadPanelFromPickle(subject, "test", True)

	if args.train:
		DoTraining(args.id, args.train)

	if args.test:
		for subject in subjects:
			DoTesting(args.id, subject)

