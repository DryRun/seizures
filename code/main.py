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
import joblib
from joblib import Parallel, delayed
import datetime
from math import sqrt
from math import ceil
import pickle as pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Location of preprocessed data
input_data_paths = {}
input_data_paths["Dog_1"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Dog_1_preprocessed.pkl"
input_data_paths["Dog_2"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Dog_2_preprocessed.pkl"
input_data_paths["Dog_4"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Dog_4_preprocessed.pkl"
input_data_paths["Dog_3"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Dog_3_preprocessed.pkl"
input_data_paths["Patient_2"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_2_preprocessed.pkl"
input_data_paths["Patient_1"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_1_preprocessed.pkl"
input_data_paths["Patient_4"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_4_preprocessed.pkl"
input_data_paths["Patient_3"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_3_preprocessed.pkl"
input_data_paths["Patient_8"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_8_preprocessed.pkl"
input_data_paths["Patient_6"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_6_preprocessed.pkl"
input_data_paths["Patient_7"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_7_preprocessed.pkl"
input_data_paths["Patient_5"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/Patient_5_preprocessed.pkl"

test_data_paths = {}
test_data_paths["Dog_1"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Dog_1_preprocessed.pkl"
test_data_paths["Dog_2"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Dog_2_preprocessed.pkl"
test_data_paths["Dog_4"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Dog_4_preprocessed.pkl"
test_data_paths["Dog_3"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Dog_3_preprocessed.pkl"
test_data_paths["Patient_2"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_2_preprocessed.pkl"
test_data_paths["Patient_1"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_1_preprocessed.pkl"
test_data_paths["Patient_4"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_4_preprocessed.pkl"
test_data_paths["Patient_3"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_3_preprocessed.pkl"
test_data_paths["Patient_8"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_8_preprocessed.pkl"
test_data_paths["Patient_6"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_6_preprocessed.pkl"
test_data_paths["Patient_7"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_7_preprocessed.pkl"
test_data_paths["Patient_5"] = "/Users/dryu/Documents/DataScience/Seizures/data/preprocessed/test_Patient_5_preprocessed.pkl"

# Train a classifier.
def TrainRandomForest(p_subject, p_save):
	print "Welcome to TrainRandomForest(" + p_subject + ", " + str(p_save) + ")"
	training_data = pd.read_pickle(input_data_paths[p_subject])

	# Ictal vs interictal
	forest_seizure = RandomForestClassifier(n_estimators = 500, n_jobs = 1, max_features="sqrt", max_depth=None, min_samples_split=1)
	y_seizure = [1 * (x > 0) for x in training_data.T["classification"]]
	forest_seizure.fit(training_data[:-2].T, y_seizure)

	# IctalA vs IctalB
	forest_early = RandomForestClassifier(n_estimators = 500, n_jobs = 1, max_features="sqrt", max_depth=None, min_samples_split=1)
	y_early = [1 * (x == 2) for x in training_data.T["classification"]]
	forest_early.fit(training_data[:-2].T, y_early)

	## Print feature importance
	#print "Seizure feature importance:"
	#seizure_feature_importance = []
	#for i in xrange(len(training_data[:-2].columns)):
	#	seizure_feature_importance.append((training_data[:-2].T.columns[i], forest_seizure.feature_importances_[i]))
	#seizure_feature_importance.sort(key=lambda tup: tup[1], reverse=True)
	#for i in xrange(min(100, len(training_data[:-2].columns))):
	#	print seizure_feature_importance[i][0] + " = " + str(seizure_feature_importance[i][1])

	#print "Early feature importance:"
	#early_feature_importance = []
	#for i in xrange(len(training_data[:-2].columns)):
	#	early_feature_importance.append((training_data[:-2].T.columns[i], forest_early.feature_importances_[i]))
	#early_feature_importance.sort(key=lambda tup: tup[1], reverse=True)
	#for i in xrange(min(100, len(training_data[:-2].columns))):
	#	print early_feature_importance[i][0] + " = " + str(early_feature_importance[i][1])

	# Save models
	if p_save:
		saved_files = joblib.dump(forest_seizure, "RF_" + p_subject + "_seizure.pkl")
		for saved_file in saved_files:
			os.system("mv " + saved_file + " /Users/dryu/Documents/DataScience/Seizures/data/models")
		saved_files = joblib.dump(forest_early, "RF_" + p_subject + "_early.pkl")
		for saved_file in saved_files:
			os.system("mv " + saved_file + " /Users/dryu/Documents/DataScience/Seizures/data/models")

	return {"seizure":forest_seizure, "early":forest_early}


def PredictRandomForest(p_subject, p_save):
	print "Welcome to PredictRandomForest(" + p_subject + ",",
	if p_save:
		print "save=True)"
	else:
		print "save=False)"

	# Load models
	forest_seizure = joblib.load("/Users/dryu/Documents/DataScience/Seizures/data/models/RF_" + p_subject + "_seizure.pkl")
	forest_early = joblib.load("/Users/dryu/Documents/DataScience/Seizures/data/models/RF_" + p_subject + "_early.pkl")

	# Load test data
	test_data = pd.read_pickle(test_data_paths[p_subject])

	# Run predictions
	predict = {}
	#test_data_transformed_seizure = forest_seizure.transform(test_data[:-2].T)
	predict_proba_seizure = forest_seizure.predict_proba(test_data[:-2].T)
	positive_index = -1
	for index in xrange(len(forest_seizure.classes_)):
		if forest_seizure.classes_[index] == 1:
			positive_index = index
	predict["seizure"] = [x[positive_index] for x in predict_proba_seizure]

	#test_data_transformed_early = forest_early.transform(test_data[:-2].T)
	predict_proba_early = forest_early.predict_proba(test_data[:-2].T)
	positive_index = -1
	for index in xrange(len(forest_early.classes_)):
		if forest_early.classes_[index] == 1:
			positive_index = index
	predict["early"] = [x[positive_index] for x in predict_proba_early]

	predict_df = pd.DataFrame(data=predict, index=test_data[:-2].columns)
	if p_save:
		# Save results
		predict_df.to_pickle("RF_" + p_subject + ".pkl")
		os.system("mv " + "RF_" + p_subject + ".pkl" + " /Users/dryu/Documents/DataScience/Seizures/data/predictions")

	return predict_df

def TrainRandomForestVariance(p_subject, p_save):
	print "Welcome to TrainRandomForestVariance(" + p_subject + ", " + str(p_save) + ")"
	training_data_raw = pd.read_pickle(input_data_paths[p_subject])
	training_data = training_data_raw[["variance" in x or "classification" in x for x in training_data_raw.index]]

	# Ictal vs interictal
	forest_seizure = RandomForestClassifier(n_estimators = 500, n_jobs = 1, max_features="sqrt", max_depth=None, min_samples_split=1)
	y_seizure = [1 * (x > 0) for x in training_data.T["classification"]]
	forest_seizure.fit(training_data[:-1].T, y_seizure)

	# IctalA vs IctalB
	forest_early = RandomForestClassifier(n_estimators = 500, n_jobs = 1, max_features="sqrt", max_depth=None, min_samples_split=1)
	y_early = [1 * (x == 2) for x in training_data.T["classification"]]
	forest_early.fit(training_data[:-1].T, y_early)

	# Save models
	if p_save:
		saved_files = joblib.dump(forest_seizure, "RFV_" + p_subject + "_seizure.pkl")
		for saved_file in saved_files:
			os.system("mv " + saved_file + " /Users/dryu/Documents/DataScience/Seizures/data/models")
		saved_files = joblib.dump(forest_early, "RFV_" + p_subject + "_early.pkl")
		for saved_file in saved_files:
			os.system("mv " + saved_file + " /Users/dryu/Documents/DataScience/Seizures/data/models")

	return {"seizure":forest_seizure, "early":forest_early}


def PredictRandomForestVariance(p_subject, p_save):
	print "Welcome to PredictRandomForest(" + p_subject + ",",
	if p_save:
		print "save=True)"
	else:
		print "save=False)"

	# Load models
	forest_seizure = joblib.load("/Users/dryu/Documents/DataScience/Seizures/data/models/RFV_" + p_subject + "_seizure.pkl")
	forest_early = joblib.load("/Users/dryu/Documents/DataScience/Seizures/data/models/RFV_" + p_subject + "_early.pkl")

	# Load test data
	test_data_raw = pd.read_pickle(test_data_paths[p_subject])
	test_data = test_data_raw[["variance" in x for x in test_data_raw.index]]

	# Run predictions
	predict = {}
	#test_data_transformed_seizure = forest_seizure.transform(test_data[:-1].T)
	predict_proba_seizure = forest_seizure.predict_proba(test_data.T)
	positive_index = -1
	for index in xrange(len(forest_seizure.classes_)):
		if forest_seizure.classes_[index] == 1:
			positive_index = index
	predict["seizure"] = [x[positive_index] for x in predict_proba_seizure]

	#test_data_transformed_early = forest_early.transform(test_data[:-1].T)
	predict_proba_early = forest_early.predict_proba(test_data.T)
	positive_index = -1
	for index in xrange(len(forest_early.classes_)):
		if forest_early.classes_[index] == 1:
			positive_index = index
	predict["early"] = [x[positive_index] for x in predict_proba_early]

	predict_df = pd.DataFrame(data=predict, index=test_data.columns)
	if p_save:
		# Save results
		predict_df.to_pickle("RFV_" + p_subject + ".pkl")
		os.system("mv " + "RFV_" + p_subject + ".pkl" + " /Users/dryu/Documents/DataScience/Seizures/data/predictions")

	return predict_df

def TrainKNeighbors(p_subject, p_save):
	print "Welcome to TrainKNeighbors(" + p_subject + ", " + str(p_save) + ")"
	training_data = pd.read_pickle(input_data_paths[p_subject])

	# Ictal vs interictal
	kneighbors = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
	y = training_data.T["classification"]
	kneighbors.fit(training_data[:-2].T, y)

	# Save models
	if p_save:
		model_save_filename = "/Users/dryu/Documents/DataScience/Seizures/data/models/KN_" + p_subject + ".pkl"
		model_save_file = open(model_save_filename, 'w')
		pickle.dump(kneighbors, model_save_file)
		model_save_file.close()

	return {"simultaneous":kneighbors}

def PredictKNeighbors(p_subject, p_save):
	print "Welcome to PredictKNeighbors(" + p_subject + ",",
	if p_save:
		print "save=True)"
	else:
		print "save=False)"

	# Load models
	model_save_file = open("/Users/dryu/Documents/DataScience/Seizures/data/models/KN_" + p_subject + ".pkl", 'r')
	kneighbors = pickle.load(model_save_file)
	model_save_file.close()

	# Load test data
	test_data = pd.read_pickle(test_data_paths[p_subject])

	# Run predictions
	predict = {}
	predict_proba = kneighbors.predict_proba(test_data[:-2].T)
	late_index = -1
	early_index = -1
	for index in xrange(len(kneighbors.classes_)):
		if kneighbors.classes_[index] == 1:
			late_index = index
		if kneighbors.classes_[index] == 2:
			early_index = index
	predict["seizure"] = [x[early_index] + x[late_index] for x in predict_proba]
	predict["early"] = [x[early_index] for x in predict_proba]

	predict_df = pd.DataFrame(data=predict, index=test_data[:-2].columns)
	if p_save:
		# Save results
		predict_df.to_pickle("KN_" + p_subject + ".pkl")
		os.system("mv " + "KN_" + p_subject + ".pkl" + " /Users/dryu/Documents/DataScience/Seizures/data/predictions")

	return predict_df

# Random forest, trying to predict ictal/early/interictal simultaneously (i.e. predict 0, 1, or 2)
def TrainRandomForestSimultaneous(p_subject, p_save):
	print "Welcome to TrainRandomForestSimultaneous(" + p_subject + ", " + str(p_save) + ")"
	training_data = pd.read_pickle(input_data_paths[p_subject])

	# Ictal vs interictal
	forest_simultaneous = RandomForestClassifier(n_estimators = 1000, n_jobs = 1, max_features="sqrt", max_depth=None, min_samples_split=1)
	y_seizure = training_data.T["classification"]
	forest_simultaneous.fit(training_data[:-2].T, y_seizure)

	# Save model
	if p_save:
		saved_files = joblib.dump(forest_simultaneous, "RFS_" + p_subject + ".pkl")
		for saved_file in saved_files:
			os.system("mv " + saved_file + " /Users/dryu/Documents/DataScience/Seizures/data/models")

	return {"simultaneous":forest_simultaneous}

def PredictRandomForestSimultaneous(p_subject, p_save):
	print "Welcome to PredictRandomForestSimultaneous(" + p_subject + ",",
	if p_save:
		print "save=True)"
	else:
		print "save=False)"

	# Load models
	forest_simultaneous = joblib.load("/Users/dryu/Documents/DataScience/Seizures/data/models/RFS_" + p_subject + ".pkl")

	# Load test data
	test_data = pd.read_pickle(test_data_paths[p_subject])

	# Run predictions
	predict = {}
	predict_proba = forest_simultaneous.predict_proba(test_data[:-2].T)
	late_index = -1
	early_index = -1
	for index in xrange(len(forest_simultaneous.classes_)):
		if forest_simultaneous.classes_[index] == 1:
			late_index = index
		if forest_simultaneous.classes_[index] == 2:
			early_index = index
	predict["seizure"] = [x[early_index] + x[late_index] for x in predict_proba]
	predict["early"] = [x[early_index] for x in predict_proba]

	predict_df = pd.DataFrame(data=predict, index=test_data[:-2].columns)
	if p_save:
		# Save results
		predict_df.to_pickle("RFS_" + p_subject + ".pkl")
		os.system("mv " + "RFS_" + p_subject + ".pkl" + " /Users/dryu/Documents/DataScience/Seizures/data/predictions")

	return predict_df


def MakeSubmission(p_method):
	output_file = open("/Users/dryu/Documents/DataScience/Seizures/data/submissions/" + p_method + "_" + datetime.datetime.now().isoformat() + ".csv", 'w')
	csvwriter = csv.writer(output_file, delimiter=',')
	csvwriter.writerow(["clip","seizure","early"])

	for subject in ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']:
		predict_df = pd.read_pickle("/Users/dryu/Documents/DataScience/Seizures/data/predictions/" + p_method + "_" + subject + ".pkl")
		for segment_name in predict_df.index:
			# Format index name (restore the .mat BS)
			segment_name_formatted = segment_name.replace("test_", subject + "_test_segment_")
			segment_name_formatted += ".mat"
			csvwriter.writerow([segment_name_formatted, round(predict_df["seizure"][segment_name], 4), round(predict_df["early"][segment_name], 4)])
	output_file.close()
	print "/Users/dryu/Documents/DataScience/Seizures/data/submissions/" + p_method + "_" + datetime.datetime.now().isoformat() + ".csv"



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Train an sklearn algorithm and/or apply it to make predictions')
	parser.add_argument('--train', type=str, help='Name of sklearn algorithm')
	parser.add_argument('--test', type=str, help='Load model and apply to test data')
	parser.add_argument('--submission', type=str, help='Format predictions for a submission')
	parser.add_argument('--subject', type=str, help='Run over a single subject only')
	parser.add_argument('--singly', action='store_true', help='Run single jobs, rather than parallel')
	args = parser.parse_args()

	# Specify subjects
	subjects = []
	if args.subject:
		subjects.append(args.subject)
	else:
		subjects = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8']

	if args.train:
		if args.train == "RF":
			Parallel(n_jobs=4)(delayed(TrainRandomForest)(subject, True) for subject in subjects)
		elif args.train == "RFV":
			Parallel(n_jobs=4)(delayed(TrainRandomForestVariance)(subject, True) for subject in subjects)
			#for subject in subjects:
			#	TrainRandomForestVariance(subject, True)
		elif args.train == "RFS":
			Parallel(n_jobs=4)(delayed(TrainRandomForestSimultaneous)(subject, True) for subject in subjects)
		elif args.train == "KN":
			for subject in subjects:
				TrainKNeighbors(subject, True)
		else:
			print "[main] ERROR : Didn't recognize training method " + args.train
			sys.exit(1)

	if args.test:
		if args.test == "RF":
			if args.singly:
				for subject in subjects:
					PredictRandomForest(subject, True)
			else:
				Parallel(n_jobs=4)(delayed(PredictRandomForest)(subject, True) for subject in subjects)
		elif args.test == "RFV":
			Parallel(n_jobs=4)(delayed(PredictRandomForestVariance)(subject, True) for subject in subjects)
		elif args.test == "RFS":
			Parallel(n_jobs=4)(delayed(PredictRandomForestSimultaneous)(subject, True) for subject in subjects)
		elif args.test == "KN":
			for subject in subjects:
				PredictKNeighbors(subject, True)
		else:
			print "[main] ERROR : Didn't recognize testing method " + args.train
			sys.exit(1)

	if args.submission:
		MakeSubmission(args.submission)


