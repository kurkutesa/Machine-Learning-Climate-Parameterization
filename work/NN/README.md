# Data Extraction

[extract_data.py](./extract_data.py): raw **DataSet** in *.nc* / *.cdf* -> var of interest **DataSet** in *.cdf*

[netcdf-flattening.py](./netcdf-flattening.py): var of interest in *.cdf* -> flattened two-dimensional (pandas) **DataFrame** in *.csv*, a direct copy of [colab/netcdf-flattening.ipynb](./colabnetcdf-flattening.ipynb) in *.py*

# Neural Networks

## Regression

 

- Code: [NN.py](./NN.py)
1. DATADIR = [ARM_1hrlater.csv](../data/forNN/ARM_1hrlater.csv)
2. train_size = 0.75
3. n_hid = [n_in, 128, 64, 32, 16, n_out]
4. run_ID = 01
5. connections = ['fc'] #, 'bn', 'do'
6. act_funcs = ['relu', 'leaky_relu']
7. loss_funcs = ['square', 'quartic']#, 'huber']
8. learning_rates = [1e-2, 1e-3, 1e-4]#, 1e-5, 1e-6]
9. plt.plot = True precipitation vs Predicted precipitation
