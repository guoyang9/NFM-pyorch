# NFM-pytorch
A pytorch implementation for Neural Factorization Machine (NFM) at SIGIR 2017. The original tensorflow implementation can be found at Xiangnan's [repo](https://github.com/hexiangnan/neural_factorization_machine.git).

#### Please download the dataset from [here](https://github.com/hexiangnan/neural_factorization_machine/tree/master/data).

## Performance Comparison
I run the model for 100 epochs and compare the performance shown in Table 3 of the original paper and keep all the settings identical with the original implementation (i.e., one hiddent layer, relu as the activation function, lr is 0.05 (should be), batch_size is 128 for frappe, 4096 for movielens).

Models		| Frappe-128 | Frappe-256 | MovieLens-128 | MovieLens-256
----------- | ---------- | ---------- | ------------- | -------------
NFM-tf		| 0.313      | 0.310      | 0.456         | 0.444
NFM-pytorch | 0.310      | 0.310      | 0.456		  | 0.446


## The requirements are as follows:
	* python==3.6
	* numpy==1.16.2
	* pytorch==1.0.1
	* tensorboardX==1.6 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)

## Example to Run:
```
python main.py --batch_size=128 --lr=0.05 --hidden_factor=128
```
