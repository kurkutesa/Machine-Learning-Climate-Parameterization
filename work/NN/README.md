# Data Extraction
[extract_data.py](./extract_data.py): raw **DataSet** in *.nc* / *.cdf* -> var of interest **DataSet** in *.cdf*

[netcdf-flattening.py](./netcdf-flattening.py): var of interest in *.cdf* -> flattened two-dimensional (pandas) **DataFrame** in *.csv*, a direct copy of [colab/netcdf-flattening.ipynb](./colabnetcdf-flattening.ipynb) in *.py*

# Neural Networks
## Regression
- Code: [NN.py](./NN.py)
1. DATADIR = [ARM_1hrlater.csv](../data/forNN/)
2. train_size = 0.75
3. n_hid = [n_in = 151, 128, 64, 32, 16, n_out = 1]
4. run_ID = [01](./log/01)
5. connections = ['fc'] #, 'bn', 'do'
6. act_funcs = ['relu', 'leaky_relu']
7. loss_funcs = ['square', 'quartic']#, 'huber']
8. learning_rates = [1e-2, 1e-3, 1e-4]#, 1e-5, 1e-6]
9. plt.plot = True precipitation vs Predicted precipitation

## Classification
- Code: [NN_cumul_class.py](./NN_cumul_class.py)
1. DATADIR = [ARM_6hrcumul.csv](../data/forNN/)
1. Classification Threshold = 0.31
2. train_size = 0.6
3. n_hid = [n_in = 151, n_out = 1]
4. run_ID = [02.0](./log/02.0); [02.1](./log/02.1); [02.2](./log/02.2)
5. connections = ['fc']
6. act_funcs = ['log_reg']#['leaky_relu','relu']
7. loss_funcs = ['xent','hinge']#,'square']
8. learning_rates = [1e-3]#, 1e-5, 1e-6]
9. plt.plot = True precipitation vs Probability of Raining

- Code: [NN_1hr_class.py](./NN_1hr_class.py)
1. DATADIR = [ARM_1hrlater.csv](../data/forNN/)
1. Classification Threshold = 0.1
2. train_size = 0.6
3. n_hid = [n_in = 151, (5), n_out = 1]
4. num_epoch = 3000
5. run_ID = [03.0](./log/03.0); [03.1](./log/03.1)
5. connections = ['fc']
7. act_funcs = ['leaky_relu','relu'] #'lr-svmlin']
8. loss_funcs = ['xent','hinge']#,'square'] - xent and hinge corresponds to logistic reg and linear SVM resp. when no hid layers
9. learning_rates = [1e-3]#, 1e-5, 1e-6]
9. plt.plot = 1D True precipitation plots for both classes separately



