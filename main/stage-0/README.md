# Sept 2018 - Jan 2019

# Summary
## Predict precipitation at ARM TWP-C1 an hour later (tried 6 hours later as well) using *in situ* measurement data
- Input data: **T_p, rh_p, u_p, v_p, prec_sfc, (t_cos, t_sin)**
- Method: NN (4 hidden layers), RF+NN
- Problem: it tries to predict most events as no precipitation events
- Suspect: current input data cannot fully capture the dynamics, and data is imbalanced (precip vs no precip, hour imbalance)

# Remarks
- All *.ipynb* files are in [colab/](./colab/).
- From Run 02.\*, run version includes a .\* at the end.
- From Run 04.\*, run version \*.0 is reserved for testground.
# Data Extraction
### Convert raw data to data of interest in NetCDF
[extract_data.py](./code/data_mani/extract_data.py): raw **DataSet** in *.nc* / *.cdf* -> var of interest **DataSet** in *.cdf*

### Convert data of interest in NetCDF to 2D (flattened) easy-to-read DataFrame-supported .csv
[netcdf-flattening.ipynb](./colab/netcdf-flattening.ipynb) <- [netcdf-flattening.py](./code/ML/netcdf-flattening.py): var of interest in *.cdf* -> flattened two-dimensional (pandas) **DataFrame** in *.csv*, append the next hour precipitation as labels

[netcdf-flattening-6-hour-cumulative-precip.ipynb](./colab/netcdf-flattening-6-hour-cumulative-precip.ipynb): ditto, but apeend the next 6-hour cumulative precipitation as labels

[RF-1hrlater.ipynb](./colab/RF-1hrlater.ipynb): append RF-predicted class onto dataset in *.csv*. Because of lack of disk quota, I cannot install more packages in the virtual environment. I have requested for more disk quota. (14 Dec 2018)

# Classical Machine Learning
## Classification - SVM - RBF kernel
### SVM classifies if it is rainy the next hour (the second best classifier)
- Code: [SVM-1hrlater.ipynb](./colab/SVM-1hrlater.ipynb)
1. DATADIR = [ARM_1hrlater.csv](../data/stage-0_cleaned/)
1. Classification Threshold = 0.1
2. train_size = 0.6
3. Rainy period ratio = 0.1659/ 0.4869 - blind test accuracy = 0.8341/ 0.5131
3. **test accuracy = 0.8922/ 0.4(bad)**
9. plt.plot = 1D True precipitation plots for both classes separately

### SVM classifies if it is rainy the next 6 hours
- Code: [SVM-6hrcumul.ipynb](./colab/SVM-6hrcumul.ipynb)
1. DATADIR = [ARM_6hrcumul.csv](../data/stage-0_cleaned/)
1. Classification Threshold = 0.3002
2. train_size = 0.6
3. Rainy period ratio = 0.4995 - blind test accuracy = 0.5
3. **test accuracy = 0.4672**
9. plt.plot = None

## Classification - Random Forest
### RF classifies if it is rainy the next hour (the best classifier)
- Code: [RF-1hrlater.ipynb](./colab/RF-1hrlater.ipynb)
1. DATADIR = [ARM_1hrlater.csv](../data/stage-0_cleaned/)
1. Classification Threshold = 0.1/ 0/ 0.05
2. train_size = 0.6
3. Rainy period ratio = 0.1659/ 0.4869/ 0.3183 - blind test accuracy = 0.8341/ 0.5131/ 0.6817
3. **test accuracy = 0.9/ 0.85/ 0.88 !!!**
9. plt.plot = 1D True precipitation plots for both classes separately

# Neural Networks
Abs loss is de-normalized, and is not used as a loss metric. Other regression losses are normalized.

## Regression
### Naïve NN regression
- Code: [NN.py](./code/ML/NN.py)
1. DATADIR = [ARM_1hrlater.csv](../data/stage-0_cleaned/)
2. train_size = 0.75
3. num_epoch = 100000
3. n_hid = [n_in = 151, 128, 64, 32, 16, n_out = 1]
4. run_ID = [01](./log/01)
5. connections = ['fc'] #, 'bn', 'do'
6. act_funcs = ['relu', 'leaky_relu']
7. loss_funcs = ['square', 'quartic']#, 'huber']
8. learning_rates = [1e-2, 1e-3, 1e-4]#, 1e-5, 1e-6]
9. plt.plot = True precipitation vs Predicted precipitation
9. ***LeakyReLU-sqloss-1e-3* mean abs loss = 1.131 < other config**, tends to all collapse to zero due to imbalanced data

### NN regression after RF binary classification
- Code: [NN_after_RF_1hr.py](./code/ML/NN_after_RF_1hr.py)
1. DATADIR = [ARM_1hrlater_RFclassified.csv](../data/stage-0_cleaned/); [ARM_1hrlater_RFclassified_threshold_0.05.csv](../data/stage-0_cleaned/)
2. train_size = 0.6 - have to follow RF config in [RF-1hrlater.ipynb](./colab/RF-1hrlater.ipynb)
3. num_epoch = 100000
3. n_hid = [n_in = 151, 128, 64, 32, 16, n_out = 1]
4. run_ID = [04.1](./log/04.1); [05.1](./log/05.1)
5. connections = ['fc']#, 'bn', 'do']
6. act_funcs = ['relu', 'leaky_relu']
1. loss_funcs = ['square', 'quartic']
1. learning_rates = [1e-3, 1e-4]
9. plt.plot = True precipitation vs Predicted precipitation in 2 colours (each for each RF class)
9. r04.1: threshold = 0, ***ReLU-sqloss-1e-3* mean abs loss = 0.8082 < other config**
9. r05.1: threshold = 0.05, ***ReLU-sqloss-1e-3* mean abs loss = 0.9153 < other config**

## Classification
### NN/ Log reg classifies if it is rainy the next 6 hours (overfit, the worst)
- Code: [NN_cumul_class.py](./code/ML/NN_cumul_class.py)
1. DATADIR = [ARM_6hrcumul.csv](../data/stage-0_cleaned/)
1. Classification Threshold = 0.31
2. train_size = 0.6
3. n_hid = [n_in = 151, n_out = 1]
4. num_epoch = 3000
4. run_ID = [02.0](./log/02.0); [02.1](./log/02.1); [02.2](./log/02.2)
5. connections = ['fc']
6. act_funcs = ['log_reg']#['leaky_relu','relu']
7. loss_funcs = ['xent','hinge']#,'square']
8. learning_rates = [1e-3]#, 1e-5, 1e-6]
9. plt.plot = True precipitation vs Probability of Raining
9. r02.0: n_hid = [n_in = 151, 16, 4, n_out = 1] ***ReLU-Hinge* accuracy = 0.5219 > other config**
9. r02.1: n_hid = [n_in = 151, 16, 4, n_out = 1] **accuracy < 0.5 sucks**
9. r02.2: ***Hinge==linSVM* accuracy = 0.5525 > 0.5473 = *xEnt==LogReg* accuracy**

### Log reg (r03.0)/ linear SVM (r03.0)/ simple 1-hid-layer NN (r03.1) classifies if it is rainy the next hour
- Code: [NN_1hr_class.py](./code/ML/NN_1hr_class.py)
1. DATADIR = [ARM_1hrlater.csv](../data/stage-0_cleaned/)
1. Classification Threshold = 0.1
2. train_size = 0.6
3. n_hid = [n_in = 151, (5), n_out = 1] - the hid layer exists in some runs only
4. num_epoch = 3000
5. run_ID = [03.0](./log/03.0); [03.1](./log/03.1)
5. connections = ['fc']
7. act_funcs = ['leaky_relu','relu'] #'lr-svmlin']
8. loss_funcs = ['xent','hinge']#,'square'] - xent and hinge corresponds to logistic reg and linear SVM resp. when no hid layers (r03.0)
9. learning_rates = [1e-3]#, 1e-5, 1e-6]
9. plt.plot = 1D True precipitation plots for both classes separately
9. r03.0: ***Hinge==linSVM* accuracy = 0.8938 > 0.8871 = *xEnt==LogReg* accuracy**
9. r03.1: ***1hd-ReLU-Hinge* accuracy = 0.8756 > other 1hd config**


