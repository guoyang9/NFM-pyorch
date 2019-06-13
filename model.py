import torch
import torch.nn as nn
import torch.nn.functional as F


class NFM(nn.Module):
	def __init__(self, num_features, num_factors, 
		act_function, layers, batch_norm, drop_prob, pretrain_FM):
		super(NFM, self).__init__()
		"""
		num_features: number of features,
		num_factors: number of hidden factors,
		act_function: activation function for MLP layer,
		layers: list of dimension of deep layers,
		batch_norm: bool type, whether to use batch norm or not,
		drop_prob: list of the dropout rate for FM and MLP,
		pretrain_FM: the pre-trained FM weights.
		"""
		self.num_features = num_features
		self.num_factors = num_factors
		self.act_function = act_function
		self.layers = layers
		self.batch_norm = batch_norm
		self.drop_prob = drop_prob
		self.pretrain_FM = pretrain_FM

		self.embeddings = nn.Embedding(num_features, num_factors)
		self.biases = nn.Embedding(num_features, 1)
		self.bias_ = torch.tensor([0.0], requires_grad=True).cuda()

		FM_modules = []
		if self.batch_norm:
			FM_modules.append(nn.BatchNorm1d(num_factors))		
		FM_modules.append(nn.Dropout(drop_prob[0]))
		self.FM_layers = nn.Sequential(*FM_modules)

		MLP_module = []
		in_dim = num_factors
		for dim in self.layers:
			out_dim = dim
			MLP_module.append(nn.Linear(in_dim, out_dim))
			in_dim = out_dim

			if self.batch_norm:
				MLP_module.append(nn.BatchNorm1d(out_dim))
			if self.act_function == 'relu':
				MLP_module.append(nn.ReLU())
			elif self.act_function == 'sigmoid':
				MLP_module.append(nn.Sigmoid())
			elif self.act_function == 'tanh':
				MLP_module.append(nn.Tanh())

			MLP_module.append(nn.Dropout(drop_prob[-1]))
		self.deep_layers = nn.Sequential(*MLP_module)

		predict_size = layers[-1] if layers else num_factors
		self.prediction = nn.Linear(predict_size, 1, bias=False)

		self._init_weight_()

	def _init_weight_(self):
		""" Try to mimic the original weight initialization. """
		if self.pretrain_FM:
			self.embeddings.weight.data.copy_(
							self.pretrain_FM.embeddings.weight)
			self.biases.weight.data.copy_(
							self.pretrain_FM.biases.weight)
			self.bias_.data.copy_(self.pretrain_FM.bias_)
		else:
			nn.init.normal_(self.embeddings.weight, std=0.01)
			nn.init.constant_(self.biases.weight, 0.0)

		# for deep layers
		if len(self.layers) > 0:
			for m in self.deep_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_normal_(m.weight)
			nn.init.xavier_normal_(self.prediction.weight)
		else:
			nn.init.constant_(self.prediction.weight, 1.0)

	def forward(self, features, feature_values):
		nonzero_embed = self.embeddings(features)
		feature_values = feature_values.unsqueeze(dim=-1)
		nonzero_embed = nonzero_embed * feature_values

		# Bi-Interaction layer
		sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
		square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

		# FM model
		FM = 0.5 * (sum_square_embed - square_sum_embed)
		FM = self.FM_layers(FM)
		if self.layers: # have deep layers
			FM = self.deep_layers(FM)
		FM = self.prediction(FM)

		# bias addition
		feature_bias = self.biases(features)
		feature_bias = (feature_bias * feature_values).sum(dim=1)
		FM = FM + feature_bias + self.bias_
		return FM.view(-1)


class FM(nn.Module):
	def __init__(self, num_features, num_factors, batch_norm, drop_prob):
		super(FM, self).__init__()
		"""
		num_features: number of features,
		num_factors: number of hidden factors,
		batch_norm: bool type, whether to use batch norm or not,
		drop_prob: list of the dropout rate for FM and MLP,
		"""
		self.num_features = num_features
		self.num_factors = num_factors
		self.batch_norm = batch_norm
		self.drop_prob = drop_prob

		self.embeddings = nn.Embedding(num_features, num_factors)
		self.biases = nn.Embedding(num_features, 1)
		self.bias_ = torch.tensor([0.0], requires_grad=True).cuda()

		FM_modules = []
		if self.batch_norm:
			FM_modules.append(nn.BatchNorm1d(num_factors))		
		FM_modules.append(nn.Dropout(drop_prob[0]))
		self.FM_layers = nn.Sequential(*FM_modules)

		nn.init.normal_(self.embeddings.weight, std=0.01)
		nn.init.constant_(self.biases.weight, 0.0)


	def forward(self, features, feature_values):
		nonzero_embed = self.embeddings(features)
		feature_values = feature_values.unsqueeze(dim=-1)
		nonzero_embed = nonzero_embed * feature_values

		# Bi-Interaction layer
		sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
		square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

		# FM model
		FM = 0.5 * (sum_square_embed - square_sum_embed)
		FM = self.FM_layers(FM).sum(dim=1, keepdim=True)
		
		# bias addition
		feature_bias = self.biases(features)
		feature_bias = (feature_bias * feature_values).sum(dim=1)
		FM = FM + feature_bias + self.bias_
		return FM.view(-1)
