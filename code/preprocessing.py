# Compute features for seizure data from raw EEG data
import scipy.io
import scipy.signal
import os
import sys
import matplotlib
import pandas as pd
import numpy as np
import random


def Preprocess(subject, feature_list):
	# Load data from pickle
	# items = segment type and number, e.g. ictal_200
	# major axis = number of measurement in the set
	# minor axis = raw features. Electrode readings + time since seizure onset.
	pickle_filename = "/Users/dryu/Documents/DataScience/Seizures/data/pickles/" + subject + "_downsampled.pkl"
	input_data = pd.read_pickle(pickle_filename) 

	for segment_name in input_data.items:
		# Columns = channel name (+time), so rows = readings
		for column_name in input_data[segment_name].columns:
			if column_name == "time":
				pass
			else:
				# Row data = readings vs. time
				


	output_data.to_pickle(subject + "_preprocessed.pkl")
	os.system("mv " + subject + "_preprocessed.pkl /Users/dryu/Documents/DataScience/Seizures/data/preprocessed")


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Preprocess ')
	parser.add_argument('subjects', type=str, help='Subject, or all to do all subjects')

	if args.subjects == "all":
		subjects = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8',]
	else:
		subjects = [args.subjects]

	for subject in subjects:
		Preprocess(subject)
