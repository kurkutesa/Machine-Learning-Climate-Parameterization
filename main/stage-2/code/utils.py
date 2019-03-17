import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from classes import Sample, SampleSubset, Scaler
seed = 42

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


def DS_mark_outliers(DS, bool_list, da_name='outliers'):
    outliers = xr.DataArray([{True: np.nan, False:1}[boolean] for boolean in bool_list], dims='time', name=da_name)
    return xr.merge([DS,outliers]).dropna(dim='time')


def get_DS_plev(DS):
    return DS['p'].values.astype(float)

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
## Data split


def split(ss, train_size, seed=seed):
    from sklearn.model_selection import train_test_split
    train_bin, test_bin, train_y, test_y, train_Xscalar, test_Xscalar, train_Xvec, test_Xvec = train_test_split(ss.bin, ss.y, ss.Xscalar, ss.Xvec,
                                                                                                                            train_size=train_size,
                                                                                                                            random_state=seed,
                                                                                                                            shuffle=True,
                                                                                                                            stratify=ss.bin)
    return SampleSubset(train_bin, train_y, train_Xscalar, train_Xvec), SampleSubset(test_bin, test_y, test_Xscalar, test_Xvec)

###################################
## Data standardization, z-score


def standardize(train, valid, test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    train = scaler.fit_transform(train)
    valid, test = scaler.transform(valid), scaler.transform(test)
    return train, valid, test, scaler


def inverse_standardize(data, scaler):
    from sklearn.preprocessing import StandardScaler
    return scaler.inverse_transform(data)


def standardize_3d(train, valid, test):
    from sklearn.preprocessing import StandardScaler
    scalers = {}
    for i in range(train.shape[2]):
        scalers[i] = StandardScaler()
        train[:, :, i] = scalers[i].fit_transform(train[:, :, i])

    for i in range(test.shape[2]):
        valid[:, :, i] = scalers[i].transform(valid[:, :, i])
        test[:, :, i] = scalers[i].transform(test[:, :, i])

    return train, valid, test, scalers


def standardize_all(s):
    train_bin, validation_bin, test_bin = s.train.bin, s.validation.bin, s.test.bin
    train_y, validation_y, test_y, scaler_y = standardize(s.train.y, s.validation.y, s.test.y)
    train_Xscalar, validation_Xscalar, test_Xscalar, scaler_Xscalar = standardize(s.train.Xscalar, s.validation.Xscalar, s.test.Xscalar)
    train_Xvec, validation_Xvec, test_Xvec, scaler_Xvec = standardize_3d(s.train.Xvec, s.validation.Xvec, s.test.Xvec)
    return Sample(SampleSubset(train_bin, train_y, train_Xscalar, train_Xvec), SampleSubset(validation_bin, validation_y, validation_Xscalar, validation_Xvec), SampleSubset(test_bin, test_y, test_Xscalar, test_Xvec), Scaler(scaler_y, scaler_Xscalar, scaler_Xvec))


def abs_zscore_cut(da, percentile):
    from scipy.stats import zscore
    return np.percentile(np.abs(zscore(da)), percentile)


def abs_zscore(da):
    from scipy.stats import zscore
    return np.abs(zscore(da))


###################################
## Data flattening (numpy)


def flattening(Xscalar, Xvec):
    X, boundary = Xscalar, []
    for i in range(0, Xvec.shape[2]):
        boundary.append(X.shape[1])
        X = np.concatenate((X, Xvec[:,:,i]), axis=1)
    return X, boundary


def flattening_all(s):
    s.train.Xflatten, s.train.Xboundary = flattening(s.train.Xscalar, s.train.Xvec)
    s.validation.Xflatten, s.validation.Xboundary = flattening(s.validation.Xscalar, s.validation.Xvec)
    s.test.Xflatten, s.test.Xboundary = flattening(s.test.Xscalar, s.test.Xvec)
    return s


def unflattening(X, boundary):
    b = boundary.copy()
    Xscalar = X[:,0:b[0]]
    b.append(X.shape[1])

    Xvec = np.expand_dims(X[:,b[0]:b[1]], axis=2)

    for i in range(1,len(b)-1):
        X_temp = np.expand_dims(X[:,b[i]:b[i+1]], axis=2)
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


###################################
## Sample concatenation


def shuffle_ss(ss):
    shuffle_index = np.random.permutation(ss.size)
    ss.bin = ss.bin[shuffle_index]
    ss.y = ss.y[shuffle_index]
    ss.Xscalar = ss.Xscalar[shuffle_index]
    ss.Xvec = ss.Xvec[shuffle_index]
    return ss


def sample_concat(s_list, shuffle=False):
    train_bin = np.concatenate(tuple([s.train.bin for s in s_list]), axis=0)
    train_y = np.concatenate(tuple([s.train.y for s in s_list]), axis=0)
    train_Xscalar = np.concatenate(tuple([s.train.Xscalar for s in s_list]), axis=0)
    train_Xvec = np.concatenate(tuple([s.train.Xvec for s in s_list]), axis=0)
    ss_train = SampleSubset(train_bin, train_y, train_Xscalar, train_Xvec)

    validation_bin = np.concatenate(tuple([s.validation.bin for s in s_list]), axis=0)
    validation_y = np.concatenate(tuple([s.validation.y for s in s_list]), axis=0)
    validation_Xscalar = np.concatenate(tuple([s.validation.Xscalar for s in s_list]), axis=0)
    validation_Xvec = np.concatenate(tuple([s.validation.Xvec for s in s_list]), axis=0)
    ss_validation = SampleSubset(validation_bin, validation_y, validation_Xscalar, validation_Xvec)

    test_bin = np.concatenate(tuple([s.test.bin for s in s_list]), axis=0)
    test_y = np.concatenate(tuple([s.test.y for s in s_list]), axis=0)
    test_Xscalar = np.concatenate(tuple([s.test.Xscalar for s in s_list]), axis=0)
    test_Xvec = np.concatenate(tuple([s.test.Xvec for s in s_list]), axis=0)
    ss_test = SampleSubset(test_bin, test_y, test_Xscalar, test_Xvec)

    if shuffle:
        ss_train, ss_validation, ss_test = [shuffle_ss(ss) for ss in [ss_train, ss_validation, ss_test]]

    return Sample(ss_train, ss_validation, ss_test)
