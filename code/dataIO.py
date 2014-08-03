import scipy.io
import scipy.signal
import os
import sys
import matplotlib
import pandas as pd
import numpy as np
import random

# Load a matlab file into a data panel
# subject = Patient_N or Dog_N
# segment_type = interictal, ictal, or test
# downsample = True or False
# train_fraction = 0 < # <1, fraction of data to split into training and internal testing. This is ignored if segment_type = test.
def LoadMAT(subject, downsample):
	dir = '/Users/dryu/Documents/DataScience/Seizures/data/clips/'+ subject + '/'
	dict = {}
	
	#load files in numerical order
	files = os.listdir(dir)
	files2 =[]
	
	for i in range(len(files)):
		qp = files[i].rfind('_') +1
		files2.append( files[i][0:qp] + (10-len(files[i][files[i].rfind('_')+1:]) )*'0' + files[i][qp:] )
    			
	#print len(files), len(files2)
	t = {key:value for key, value in zip(files2,files)}
	files2 = t.keys()
	files2.sort()
	f = [t[i] for i in files2]
	
	j = 0
	for i in f:
			
		seg = i[i.rfind('_')+1 : i.find('.mat')] # Number of segment, e.g. Dog_1_interictal_segment_250.mat => 250
		segtype = i[i[0:i.find('_segment')].rfind('_')+1: i.find('_segment')] # Type of segment: ictal, interictal, test
		d = scipy.io.loadmat(dir+i)
		if j==0:
			cols = range(len(d['channels'][0,0]))
			cols = cols +['time']

		if  segtype == 'interictal' or segtype == "test":
			l = -3600.0#np.nan
		else:
			#print i
			l = d['latency'][0]
			
		df = pd.DataFrame(np.append(d['data'].T, l+np.array([range(len(d['data'][1]))]).T/d['freq'][0], 1 ), index=range(len(d['data'][1])), columns=cols)
		
		if downsample:
			if np.round(d['freq'][0]) == 5000:
				df = df.groupby(lambda x: int(np.floor(x/20.0))).mean()
			if np.round(d['freq'][0]) == 500:
				df = df.groupby(lambda x: int(np.floor(x/2.0))).mean()	
			if np.round(d['freq'][0]) == 400:
				df = df.groupby(lambda x: int(np.floor(x/2.0))).mean()					
		
			df['time'] = df['time'] - (df['time'][0]-np.floor(df['time'][0]))*(df['time'][0] > 0)
		
		dict.update({segtype+'_'+seg : df})

		j = j +1
			
	data = pd.Panel(dict)
	return data

def MATToPickle(subject, downsample):
	print "Welcome to MATToPickle(" + subject + ", ",
	if downsample:
		print "True",
	else:
		print "False",
	print ")"

	pickle_directory = "/Users/dryu/Documents/DataScience/Seizures/data/pickles/"
	pickle_filename = subject
	if downsample:
		pickle_filename += "_downsampled"
	pickle_filename = pickle_filename + ".pkl"

	SavePanelAsPickle(LoadMAT(subject, downsample), pickle_filename)

def SavePanelAsPickle(data, pickle_filename):
	data.to_pickle(pickle_filename)

def LoadPanelFromPickle(subject, downsample):
	pickle_directory = "/Users/dryu/Documents/DataScience/Seizures/data/pickles/"
	pickle_filename = subject
	if downsample:
		pickle_filename += "_downsampled"
	pickle_filename += ".pkl"
	return pd.read_pickle(pickle_filename)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Process input data into pandas pickles')
	parser.add_argument('subjects', type=str, help='Subject, or all to do all subjects')
	parser.add_argument('--downsample', action='store_true', help='Downsample data')
	args = parser.parse_args()

	if args.subjects == "all":
		subjects = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8',]
	else:
		subjects = [args.subjects]

	for subject in subjects:
		MATToPickle(subject, args.downsample)



