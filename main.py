import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.05, 
	help="learning rate")
parser.add_argument("--dropout", 
	default='[0.5, 0.2]',  
	help="dropout rate for FM and MLP")
parser.add_argument("--batch_size", 
	type=int, 
	default=128, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=100, 
	help="training epochs")
parser.add_argument("--hidden_factor", 
	type=int,
	default=64, 
	help="predictive factors numbers in the model")
parser.add_argument("--layers", 
	default='[64]', 
	help="size of layers in MLP model, '[]' is NFM-0")
parser.add_argument("--lamda", 
	type=float, 
	default=0.0, 
	help="regularizer for bilinear layers")
parser.add_argument("--batch_norm", 
	default=True, 
	help="use batch_norm or not")
parser.add_argument("--pre_train", 
	action='store_true', 
	default=False, 
	help="whether use the pre-train or not")
parser.add_argument("--out", 
	default=True, 
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


#############################  PREPARE DATASET #########################
features_map, num_features = data_utils.map_features()

train_dataset = data_utils.FMData(config.train_libfm, features_map)
valid_dataset = data_utils.FMData(config.valid_libfm, features_map)
test_dataset = data_utils.FMData(config.test_libfm, features_map)

train_loader = data.DataLoader(train_dataset, drop_last=True,
			batch_size=args.batch_size, shuffle=True, num_workers=4)
valid_loader = data.DataLoader(valid_dataset,
			batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader = data.DataLoader(test_dataset,
			batch_size=args.batch_size, shuffle=False, num_workers=0)

##############################  CREATE MODEL ###########################
if args.pre_train:
	assert os.path.exists(config.FM_model_path), 'lack of FM model'
	assert config.model == 'NFM', 'only support NFM for now'
	FM_model = torch.load(config.FM_model_path)
else:
	FM_model = None

if config.model == 'FM':
	model = model.FM(num_features, args.hidden_factor,
					args.batch_norm, eval(args.dropout))
else:
	model = model.NFM(
		num_features, args.hidden_factor, 
		config.activation_function, eval(args.layers), 
		args.batch_norm, eval(args.dropout), FM_model)
model.cuda()
if config.optimizer == 'Adagrad':
	optimizer = optim.Adagrad(
		model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
elif config.optimizer == 'Adam':
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif config.optimizer == 'SGD':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif config.optimizer == 'Momentum':
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

if config.loss_type == 'square_loss':
	criterion = nn.MSELoss(reduction='sum')
else: # log_loss
	criterion = nn.BCEWithLogitsLoss(reduction='sum')

# writer = SummaryWriter() # for visualization

###############################  TRAINING ############################
count, best_rmse = 0, 100
for epoch in range(args.epochs):
	model.train() # Enable dropout and batch_norm
	start_time = time.time()

	for features, feature_values, label in train_loader:
		features = features.cuda()
		feature_values = feature_values.cuda()
		label = label.cuda()

		model.zero_grad()
		prediction = model(features, feature_values)
		loss = criterion(prediction, label) 
		loss += args.lamda * model.embeddings.weight.norm()
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		count += 1

	model.eval()
	train_result = evaluate.metrics(model, train_loader)
	valid_result = evaluate.metrics(model, valid_loader)
	test_result = evaluate.metrics(model, test_loader)

	print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime(
						"%H: %M: %S", time.gmtime(time.time()-start_time)))
	print("Train_RMSE: {:.3f}, Valid_RMSE: {:.3f}, Test_RMSE: {:.3f}".format(
						train_result, valid_result, test_result))

	if test_result < best_rmse:
		best_rmse, best_epoch = test_result, epoch
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model, 
				'{}{}.pth'.format(config.model_path, config.model))

print("End. Best epoch {:03d}: Test_RMSE is {:.3f}".format(best_epoch, best_rmse))
