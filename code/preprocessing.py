# Compute features for seizure data from raw EEG data










if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description = 'Preprocess ')
	parser.add_argument('subjects', type=str, help='Subject, or all to do all subjects')
