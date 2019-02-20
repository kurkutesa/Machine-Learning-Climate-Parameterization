import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

###################################
## DS operations

# e.g. shift precipitation to an hour early
def DS_shift_and_append(DS, var_name, new_var_name, shift_hour=1):
    if shift_hour < 0:
        raise ValueError('shift_hour < 0 is not appropriate for prediction.')
    else:
        da_shifted = DS[var_name].shift(time=-shift_hour).rename(new_var_name)
        DS_shifted = xr.merge([da_shifted, DS])
        return DS_shifted


# extract variables of interest from DS
def DS_extract(DS, extract_list, drop_list=None):
    DS_select = DS[extract_list]
    for var in drop_list:
        DS_select = DS_select.drop(var)
    return DS_select


# count # valid instances in each var
def DS_count_valid(DS, dim='time'):
    cnt = []
    for var in list(DS):
        try:
            cnt.append(DS[var].dropna(dim=dim)[dim].size)
        except:
            cnt.append(0)
    return pd.DataFrame(data={'var': list(DS), '#valid': cnt}).set_index('var')


# flatten DS to DS (used to be obsolete, but still use for plotting correlations)
def DS_flatten(DS, str_1d, str_2d):
    plev = DS['p'].values.astype(np.int32)
    DS_flattened = xr.Dataset()

    for var_str in str_1d:
        DS_flattened = xr.merge([DS_flattened, DS[var_str]])

    for var_str in str_2d:
        for _p in plev:
            new_var_str = f'{var_str}{_p}'
            DS_flattened = xr.merge(
                [DS_flattened, DS[var_str].sel(p=_p).drop('p').rename(new_var_str)])

    return DS_flattened

###################################
## Data extraction (DS2df)


def extract_scalar(DS, str_y, str_Xscalar):
    return DS[str_y].to_dataframe().values, DS[str_Xscalar].to_dataframe().values


def merge_channels(DS, str_Xvec):
    channels = [DS[str_Xvec[i]].to_dataframe().unstack(level=-1)
                for i in range(0, len(str_Xvec))]
    X_conv = np.expand_dims(channels[0].values, axis=2)

    for channel in channels[1:]:
        channel = np.expand_dims(channel.values, axis=2)
        X_conv = np.append(X_conv, channel, axis=2)

    return X_conv

###################################
## Data standardization


def standardize(train, test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test, scaler


def standardize_3d(train, test):
    from sklearn.preprocessing import StandardScaler
    scalers = {}
    for i in range(train.shape[2]):
        scalers[i] = StandardScaler()
        train[:, :, i] = scalers[i].fit_transform(train[:, :, i])

    for i in range(test.shape[2]):
        test[:, :, i] = scalers[i].transform(test[:, :, i])

    return train, test, scalers

###################################
## Data flattening (numpy)


def flattening(Xscalar, Xvec):
    X, boundary = Xscalar, []
    for i in range(0, Xvec.shape[2]):
        boundary.append(X.shape[1])
        X = np.concatenate((X, Xvec[:,:,i]), axis=1)
    return X, boundary


def unflattening(X, boundary):
    Xscalar = X[:,0:boundary[0]]
    boundary.append(X.shape[1])

    Xvec = np.expand_dims(X[:,boundary[0]:boundary[1]], axis=2)

    for i in range(1,len(boundary)-1):
        X_temp = np.expand_dims(X[:,boundary[i]:boundary[i+1]], axis=2)
        Xvec = np.append(Xvec, X_temp, axis=2)

    return Xscalar, Xvec

###################################
## Data de-concatenation (numpy)


def all_x(data):
    train_X, test_X, train_y, test_y = data
    return np.concatenate((train_X, test_X))


def all_y(data):
    train_X, test_X, train_y, test_y = data
    return np.concatenate((train_y, test_y))

###################################
## DS Plot


def plot_1d(DS, var_1d):
    time_value = DS.time.values
    for var_str in var_1d:
        x_value = DS[var_str].values
        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))

        sns.scatterplot(x_value, time_value, s=3, ax=ax[0])
        ax[0].set(xlabel=var_str, ylabel='Year')

        sns.distplot(x_value[~np.isnan(x_value)]) # distplot cannot handle NaN itself
        ax[1].set(xlabel=var_str, ylabel='Frequency')

        plt.show()
    return None


def plot_2d(DS, var_2d):
    for var_str in var_2d:
        x_value = DS[var_str].values
        fig, ax = plt.subplots(ncols=2, figsize=(20, 10))

        DS[var_str].plot(ax=ax[0])

        sns.distplot(x_value[~np.isnan(x_value)]) # distplot cannot handle NaN itself
        ax[1].set(xlabel=var_str, ylabel='Frequency')

        plt.show()
    return None


def binplot_prec(DS, bin_min):
    x_value = DS['prec_sfc'].values
    x_value = x_value[x_value>=bin_min]

    fig, ax = plt.subplots(ncols=1, figsize=(10, 10))

    sns.distplot(x_value[~np.isnan(x_value)]) # distplot cannot handle NaN itself
    ax.set(xlabel='prec_sfc', ylabel='Frequency')

    plt.show()
    return None

###################################
## df Plot


def plot_pair(df, cols_str, hue_str=None):
    sns_plot = sns.pairplot(df,
                            vars=cols_str,
                            hue=hue_str,
                            palette='bright',
                            diag_kind='kde',
                            plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                            height=4)
    plt.show()
    return None

def plot_corr(df, annotate=False):
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))

    sns_plot = sns.heatmap(corr,
                           mask=mask,
                           annot=annotate,
                           vmax=.5,
                           center=0,
                           square=True,
                           linewidths=.5,
                           cbar_kws={"shrink": .5})

    plt.title('Correlation matrix', fontsize=15)
    plt.show()
    return None
