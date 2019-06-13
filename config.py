# dataset name
dataset = 'frappe'
assert dataset in ['ml-tag', 'frappe']

# model name
model = 'NFM'
assert model in ['FM', 'NFM']

# as the log_loss is not implemented in Xiangnan's repo, I thus remove this loss type
loss_type = 'square_loss'
assert loss_type in ['square_loss'] 

# important settings (normally default is the paper choice)
optimizer = 'Adagrad'
activation_function = 'relu'
assert optimizer in ['Adagrad', 'Adam', 'SGD', 'Momentum']
assert activation_function in ['relu', 'sigmoid', 'tanh', 'identity']

# paths
main_path = '/home/share/guoyangyang/recommendation/NFM-Data/{}/'.format(dataset)

train_libfm = main_path + '{}.train.libfm'.format(dataset)
valid_libfm = main_path + '{}.validation.libfm'.format(dataset)
test_libfm = main_path + '{}.test.libfm'.format(dataset)

model_path = './models/'
FM_model_path = model_path + 'FM.pth' # useful when pre-train
NFM_model_path = model_path + 'NFM.pth'
