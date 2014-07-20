import os
import sys
import numpy as np
import scipy as sp
import scipy.io as spio
import sklearn
from sklearn import svm
import glob
import EEGFunctions
import csv

topfolder = "/Users/dryu/Documents/DataScience/Seizures/"

# Setup data
data_topfolder = "/Users/dryu/Documents/DataScience/Seizures/data/"
#subjects = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5", "Patient_6", "Patient_7", "Patient_8"] # Full list
#subjects = ["Patient_4"] # One subject, for development purposes
subjects = ["Patient_5", "Patient_6", "Patient_7", "Patient_8"] # Half list, because Patient_4 was causing problems

# Get number of segments for each patient 
nsegments = {}
for subject in subjects:
	nsegments[subject] = {}
	for segment_type in ["ictal", "interictal", "test"]:
		nsegments[subject][segment_type] = len(glob.glob(data_topfolder + "/clips/" + subject + "/*_" + segment_type + "_*"))

# Train 
for subject in subjects:
	print "On subject " + subject
	# Make X and y
	n_samples = nsegments[subject]["ictal"] + nsegments[subject]["interictal"]
	# Look in a file to get the number of features
	example_segment_contents = spio.loadmat(glob.glob(data_topfolder + "/clips/" + subject + "/*interictal*")[0])
	n_features = len(example_segment_contents["data"]) * 4 # Variance and 3 moments
	X = {}
	y_isseizure = {}
	y_latency = {}
	y_combined = {} # 0 = interictal, 1 = ictal >15s, 2 = ictal < 15s
	for segment_type in ["ictal", "interictal"]:
		X[segment_type] = np.ndarray((nsegments[subject][segment_type], n_features))
		y_isseizure[segment_type] = np.ndarray((nsegments[subject][segment_type]))
		y_latency[segment_type] = np.ndarray((nsegments[subject][segment_type]))
		y_combined[segment_type] = np.ndarray((nsegments[subject][segment_type]))

		# Start loop over segments
		counter = 0
		n_segments = len(glob.glob(data_topfolder + "/clips/" + subject + "/*_" + segment_type + "_*"))
		sample_number = 0
		for segment_filename in glob.glob(data_topfolder + "/clips/" + subject + "/*_" + segment_type + "_*"):
			if counter % 100 == 0:
				print "\tProcessing " + segment_type + " training segment " + str(counter) + " / " + str(n_segments)
			counter += 1
			segment_contents = spio.loadmat(segment_filename)
			segment_eeg_data = segment_contents["data"]

			# Loop over electrodes
			feature_number = 0
			for electrode_number in xrange(len(segment_eeg_data)):
				electrode_data = segment_eeg_data[electrode_number]

				# Variance
				X[segment_type][sample_number][feature_number] = EEGFunctions.Variance(electrode_data)
				feature_number += 1
				# Moments
				X[segment_type][sample_number][feature_number] = EEGFunctions.Moment(electrode_data, 2)
				feature_number += 1
				X[segment_type][sample_number][feature_number] = EEGFunctions.Moment(electrode_data, 3)
				feature_number += 1
				X[segment_type][sample_number][feature_number] = EEGFunctions.Moment(electrode_data, 4)
				feature_number += 1

			if segment_type == "ictal":
				y_isseizure[segment_type][sample_number] = 1
				if segment_contents["latency"] <= 14:
					y_latency[segment_type][sample_number] = 1
				else:
					y_latency[segment_type][sample_number] = 0
			else:
				y_isseizure[segment_type][sample_number] = 0
				y_latency[segment_type][sample_number] = 0
			if segment_type == "ictal":
				if segment_contents["latency"] <= 14:
					y_combined[segment_type][sample_number] = 2
				else:
					y_combined[segment_type][sample_number] = 1
			else:
				y_combined[segment_type][sample_number] = 0

			sample_number += 1
		# End loop over segment of a particular type
	# End loop over segment types

	# Make SVC
	print "Starting LinearSVC classification"
	#classifier_isseizure = svm.LinearSVC()
	#classifier_isseizure.fit(np.concatenate((X["ictal"], X["interictal"])), np.concatenate((y_isseizure["ictal"], y_isseizure["interictal"])))

	#classifier_latency = svm.LinearSVC()
	#classifier_latency.fit(X["ictal"], y_latency["ictal"])

	classifier_combined = svm.LinearSVC()
	classifier_combined.fit(np.concatenate((X["ictal"], X["interictal"])), np.concatenate((y_combined["ictal"], y_combined["interictal"])))

	# Load testing data
	X_test = np.ndarray((nsegments[subject]["test"], n_features))
	X_filenames = []
	sample_number = 0
	test_samples = glob.glob(data_topfolder + "/clips/" + subject + "/*_test_*")
	n_test_samples = len(test_samples)
	counter = 0
	for segment_filename in test_samples:
		if counter % 100 == 0:
			print "\tProcessing test segment " + str(counter) + " / " + str(n_test_samples)
		counter += 1
		segment_contents = spio.loadmat(segment_filename)
		segment_eeg_data = segment_contents["data"]

		# Loop over electrodes
		feature_number = 0
		for electrode_number in xrange(len(segment_eeg_data)):
			electrode_data = segment_eeg_data[electrode_number]

			# Variance
			X_test[sample_number][feature_number] = EEGFunctions.Variance(electrode_data)
			feature_number += 1
			# Moments
			X_test[sample_number][feature_number] = EEGFunctions.Moment(electrode_data, 2)
			feature_number += 1
			X_test[sample_number][feature_number] = EEGFunctions.Moment(electrode_data, 3)
			feature_number += 1
			X_test[sample_number][feature_number] = EEGFunctions.Moment(electrode_data, 4)
			feature_number += 1
		X_filenames.append(os.path.basename(segment_filename))
		sample_number += 1
	# End loop over test segments
	
	# Run prediction
	#prediction_isseizure = classifier_isseizure.predict(X_test)
	#prediction_latency = classifier_latency.predict(X_test)
	prediction_combined = classifier_combined.predict(X_test)

	# Save as CSV
	output_file = open(topfolder + "/Results/" + subject + ".csv", 'w')
	csvwriter = csv.writer(output_file, delimiter=',')
	csvwriter.writerow(["clip","seizure"])
	for i in xrange(len(prediction_combined)):
		if prediction_combined[i] == 2:
			csvwriter.writerow([X_filenames[i], 1, 1])
		elif prediction_combined[i] == 1:
			csvwriter.writerow([X_filenames[i], 1, 0])
		else:
			csvwriter.writerow([X_filenames[i], 0, 0])
	output_file.close()
# End loop over subjects


