import os
import sys
import numpy as np
import scipy as sp
import sklearn
import glob

# Setup data
data_topfolder = "/Users/dryu/Documents/DataScience/Seizures/data/"
# subjects = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5", "Patient_6", "Patient_7", "Patient_8"] # Full list
subjects = ["Dog_1"] # One subject, for development purposes

# Get number of segments for each patient 
nsegments = {}
for subject in subjects:
	nsegments[subject] = {}
	for segment_type in ["ictal", "interictal", "test"]:
		nsegments[subject][segment_type] = len(glob.glob(data_topfolder + "/clips/" + subject + "/*" + segment_type + "*"))