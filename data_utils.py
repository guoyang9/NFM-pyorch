import numpy as np
import torch.utils.data as data

import config


def read_features(file, features):
	""" Read features from the given file. """
	i = len(features)
	with open(file, 'r') as fd:
		line = fd.readline()
		while line:
			items = line.strip().split()
			for item in items[1:]:
				item = item.split(':')[0]
				if item not in features:
					features[item] = i
					i += 1
			line = fd.readline()
	return features


def map_features():
	""" Get the number of existing features in all the three files. """
	features = {}
	features = read_features(config.train_libfm, features)
	features = read_features(config.valid_libfm, features)
	features = read_features(config.test_libfm, features)
	print("number of features: {}".format(len(features)))
	return features, len(features)


class FMData(data.Dataset):
	""" Construct the FM pytorch dataset. """
	def __init__(self, file, feature_map):
		super(FMData, self).__init__()
		self.label = []
		self.features = []
		self.feature_values = []

		with open(file, 'r') as fd:
			line = fd.readline()

			while line:
				items = line.strip().split()

				# convert features
				raw = [item.split(':')[0] for item in items[1:]]
				self.features.append(
					np.array([feature_map[item] for item in raw]))
				self.feature_values.append(np.array(
					[item.split(':')[1] for item in items[1:]], dtype=np.float32))

				# convert labels
				if config.loss_type == 'square_loss':
					self.label.append(np.float32(items[0]))
				else: # log_loss
					label = 1 if float(items[0]) > 0 else 0
					self.label.append(label)

				line = fd.readline()

		assert all(len(item) == len(self.features[0]
			) for item in self.features), 'features are of different length'

	def __len__(self):
		return len(self.label)

	def __getitem__(self, idx):
		label = self.label[idx]
		features = self.features[idx]
		feature_values = self.feature_values[idx]
		return features, feature_values, label
