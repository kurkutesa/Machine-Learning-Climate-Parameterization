# Jan 2019 - present

# Neural Networks
Abs loss is de-normalized, and is not used as a loss metric. Other regression losses are normalized.

## Regression
### NN regression
- Code: [regression.py](./code/ML/regression.py)
1. DATADIR = [twparmbeatmC1_no_nan.csv](../data/stage-1_cleaned/)
2. train_size = 0.70
3. num_epoch = 100000
3. n_hid = [n_in = 151, (16), n_out = 1] (hidden layer only exist in run 11.4)
4. run_ID = [11.3](./log/11.3), [11.4] (./log/11.4)
5. connections = ['fc']
6. act_funcs = ['relu']
7. loss_funcs = ['square_l2']
8. learning_rates = [1e-3]
9. beta = 0.01
9. plt.plot = True precipitation vs Predicted precipitation
9. Mean abs loss = 1.24 (run 11.3), 1.47 (run 11.4), same old collapse-to-zero phenomenon
# Remarks
- All *.ipynb* files are in [colab/](./colab/).
