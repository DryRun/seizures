import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import glob
import joblib
from joblib import Parallel, delayed

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import seaborn as sns

import math

import re
pattern_segment_number = re.compile("(?P<segment_number>\d+)")

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def GetVariance(p_list):
	np_list = np.array(p_list)
	return np.var(np_list[~is_outlier(np_list)])

def PlotAllElectrodes(p_subject):
	print "Welcome to PlotAllElectrodes(" + p_subject + ")"

	# Load data from pickle
	# items = segment type and number, e.g. ictal_200
	# major axis = number of measurement in the set
	# minor axis = raw features. Electrode readings + time since seizure onset.
	pickle_filename = "/Users/dryu/Documents/DataScience/Seizures/data/pickles/" + p_subject + "_downsampled.pkl"
	input_data = pd.read_pickle(pickle_filename) 

	# Loop over segments and electrodes
	print "Loading data"
	electrodes = []
	#electrode_data = {"ictal":{}, "interictal":{}, "test":{}}
	electrode_times = {"ictal":{}, "interictal":{}, "test":{}}
	electrode_readings = {"ictal":{}, "interictal":{}, "test":{}}
	for electrode_name in input_data.minor_axis:
		if electrode_name == "time":
			continue
		electrodes.append(electrode_name)
		electrode_times["ictal"][electrode_name] = []
		electrode_times["interictal"][electrode_name] = []
		electrode_times["test"][electrode_name] = []
		electrode_readings["ictal"][electrode_name] = []
		electrode_readings["interictal"][electrode_name] = []
		electrode_readings["test"][electrode_name] = []

		for segment_name in input_data.items:
			if "interictal" in segment_name:
				segment_type = "interictal"
			elif "ictal" in segment_name:
				segment_type = "ictal"
			else:
				segment_type = "test"

			# Time offset: for test and interictal segments, the time offset is the segment number + 3600 (-3600 was applied in dataIO.py)
			time_offset = 0
			if segment_type == "interictal" or segment_type == "test":
				# Regex the segment #
				match_segment_number = pattern_segment_number.search(segment_name)
				if match_segment_number == None:
					print "ERROR : Couldn't regex the segment number out of " + segment_name
					sys.exit(1)
				time_offset = int(match_segment_number.group("segment_number")) + 3600

			#electrode_data[segment_type][electrode_name].extend(zip(input_data[segment_name]["time"] + time_offset, input_data[segment_name][electrode_name]))
			electrode_times[segment_type][electrode_name].extend(input_data[segment_name]["time"] + time_offset)
			electrode_readings[segment_type][electrode_name].extend(input_data[segment_name][electrode_name])

	# Turn the times * readings into a series. Also, compute variances. Also, normalize electrode readings to 2*variance.
	print "Formatting data"
	electrode_data = {}
	variances = {}
	n_variances_plot = 1
	for segment_type in ["ictal", "interictal", "test"]:
		electrode_data[segment_type] = {}
		variances[segment_type] = {}
		vertical_offset = 0
		for electrode_name in electrodes:
			print "\t" + segment_type + " / " + str(electrode_name)
			#print electrode_readings[segment_type][electrode_name]
			variances[segment_type][electrode_name] = GetVariance(electrode_readings[segment_type][electrode_name])
			electrode_data[segment_type][electrode_name] = pd.Series(electrode_readings[segment_type][electrode_name], index=electrode_times[segment_type][electrode_name])
			print "\t\tVariance=" + str(variances[segment_type][electrode_name])
			#electrode_data[segment_type][electrode_name] = electrode_data[segment_type][electrode_name]	/ variances[segment_type][electrode_name]
			electrode_data[segment_type][electrode_name] = electrode_data[segment_type][electrode_name]	+ vertical_offset
			vertical_offset += variances[segment_type][electrode_name]

	# Start plotting
	print "Plotting"
	max_seconds_per_plot = 5
	for segment_type in ["ictal", "interictal", "test"]:
		start_time = min(electrode_times[segment_type][electrode_name])
		end_time = max(electrode_times[segment_type][electrode_name])
		n_plots = int(math.ceil((end_time - start_time) / max_seconds_per_plot))
		intervals = []
		for i in xrange(n_plots):
			intervals.append((start_time + i * max_seconds_per_plot, start_time + (i+1) * max_seconds_per_plot))
		for i in xrange(n_plots):
			# Plot width: 2 inches per second
			width = 2 * max_seconds_per_plot
			figure = plt.figure(figsize=(width, 6))
			for electrode_name in electrodes:
				electrode_data[segment_type][electrode_name].plot()
			figure.get_axes()[0].set_xlim(intervals[i][0], intervals[i][1])
			figure_directory = "figures/EEGs/" + p_subject
			os.system("mkdir -pv " + figure_directory)
			pyl.savefig(figure_directory + "/" + segment_type + "_" + str(int(intervals[i][0])) + "-" + str(int(intervals[i][1])) + ".png")
			plt.close()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Visualize EEGs')
	parser.add_argument('subjects', type=str, help='Subject, or all to do all subjects')
	args = parser.parse_args()

	if args.subjects == "all":
		subjects = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8',]
	else:
		subjects = [args.subjects]

	for subject in subjects:
		PlotAllElectrodes(subject)
	#Parallel(n_jobs=4)(delayed(PlotAllElectrodes)(subject) for subject in subjects)

	print "All done."
