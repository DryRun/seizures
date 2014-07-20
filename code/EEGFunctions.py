import numpy as np
import scipy.stats.mstats as scimom

def Variance(data):
	return np.var(data)

def Moment(data, moment_n):
	return scimom.moment(data, moment_n)

