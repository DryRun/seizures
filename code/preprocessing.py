# Compute features for seizure data from raw EEG data
import scipy.io
import scipy.signal
import os
import sys
import matplotlib
import pandas as pd
import numpy as np
import random
import scipy.fftpack as fft
from math import *
from joblib import Parallel, delayed


def Preprocess(subject):
	print "Welcome to Preprocess(" + subject + ")"
	# Load data from pickle
	# items = segment type and number, e.g. ictal_200
	# major axis = number of measurement in the set
	# minor axis = raw features. Electrode readings + time since seizure onset.
	pickle_filename = "/Users/dryu/Documents/DataScience/Seizures/data/pickles/" + subject + "_downsampled.pkl"
	input_data = pd.read_pickle(pickle_filename) 
	output_data_dict = {}
	output_test_dict = {}

	first = True
	counter = 0
	for segment_name in input_data.items:
		if counter % 1000 == 0:
			print "\tSegment " + str(counter) + " / " + str(len(input_data.items))
		counter += 1
		if first:
			first = False
			# Make list of features
			features = []
			for column_name in input_data[segment_name].columns:
				if column_name == "time":
					continue
				features.append("electrode" + str(column_name) + "_variance")
				n_samples = len(input_data[segment_name][column_name])
				sampling_frequency = n_samples / 1.
				fft_length = int(2**(ceil(log(n_samples) / log(2))))
				for i in xrange((fft_length / 2) + 1):
					freq = 1. * i * sampling_frequency / fft_length
					features.append("electrode" + str(column_name) + "_f" + str(round(freq, 2)) + "Hz")	
			features.append("latency")
			features.append("classification") # 0 = interictal, 1 = ictalB, 3 = ictalA

		segment_features = pd.Series(index=features)

		# Columns = channel name (+time), so rows = readings
		for column_name in input_data[segment_name].columns:
			if column_name == "time":
				continue
			else:
				# Variance
				segment_features["electrode" + str(column_name) + "_variance"] = np.var(input_data[segment_name][column_name])

				# Row data = readings vs. time
				n_samples = len(input_data[segment_name][column_name])
				sampling_frequency = n_samples / 1. # Always 1 second of data
				fft_length = int(2**(ceil(log(n_samples) / log(2))))
				ft = fft.rfft(input_data[segment_name][column_name], fft_length) # [y(0),Re(y(1)),Im(y(1)),...,Re(y(n/2))]
				ft_power = []

				total_power = 0.
				for i in xrange((fft_length / 2) + 1):
					total_power += ft[i]**2

				for i in xrange((fft_length / 2) + 1):
					freq = 1. * i * sampling_frequency / fft_length
					if i == 0 or i == fft_length / 2:
						power = ft[i]**2
					else:
						power = ft[2*i]**2 + ft[(2*i)-1]**2
					segment_features["electrode" + str(column_name) + "_f" + str(round(freq, 2)) + "Hz"] = power / total_power
				# End loop of FFT frequencies
			# End if-else loop rejecting "time"
		# End loop over electrodes
		segment_latency = input_data[segment_name]["time"][0]
		segment_features["latency"] = input_data[segment_name]["time"][0]

		if "test" in segment_name:
			segment_features["classification"] = -1
		if segment_latency < 0:
			segment_features["classification"] = 0
		elif segment_latency <= 15:
			segment_features["classification"] = 2
		else:
			segment_features["classification"] = 1

		if "test" in segment_name:
			output_test_dict[segment_name] = segment_features
		else:
			output_data_dict[segment_name] = segment_features
	output_data = pd.DataFrame(output_data_dict)
	output_test = pd.DataFrame(output_test_dict)


	output_data.to_pickle(subject + "_preprocessed.pkl")
	os.system("mv " + subject + "_preprocessed.pkl /Users/dryu/Documents/DataScience/Seizures/data/preprocessed")
	output_test.to_pickle("test_" + subject + "_preprocessed.pkl")
	os.system("mv test_" + subject + "_preprocessed.pkl /Users/dryu/Documents/DataScience/Seizures/data/preprocessed")


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Preprocess ')
	parser.add_argument('subjects', type=str, help='Subject, or all to do all subjects')
	args = parser.parse_args()

	if args.subjects == "all":
		subjects = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8',]
	else:
		subjects = [args.subjects]

	Parallel(n_jobs=4)(delayed(Preprocess)(subject) for subject in subjects)
	#for subject in subjects:
	#	Preprocess(subject)
