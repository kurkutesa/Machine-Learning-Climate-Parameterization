# For reference Only
# Calculate CAPE from single column with different HEIGHT (z) level
# Embed surface value into column
def embed_sfc_to_col_dropna(da_sfc, DS_col, sfc_elevation=2.0):
    return xr.concat([da_sfc.assign_coords(z=np.float32(sfc_elevation)), DS_col], dim='z').drop(['z10', 'z2']).dropna(dim='time')


def CAPE_param(T_z_DS, env_lapse_rate=6.5, sfc_elevation=2.0):  # called by other functions only
    T_ap = list(T_z_DS.values)

    T_env = []
    T_sfc = T_z_DS.sel(z=sfc_elevation).values
    for z in T_z_DS['z']:
        GAMMA = env_lapse_rate / 1000  # convert it from per km to per m
        t = T_sfc - (z-sfc_elevation) * GAMMA
        T_env.append(float(t))

    delta_z = np.diff(T_z_DS['z'].values)
    z_profile = T_z_DS['z'].values
    return T_ap, T_env, delta_z, z_profile


def CAPE_discretize(T_ap, T_env, delta_z):    # called by other functions only
    # for every delta_z, ignore constant term g:
    B = [(x-y)/y for x, y in zip(T_ap, T_env)]
    delta_z = np.concatenate((delta_z, [0]))

    delta_CAPE = B * delta_z
    return delta_CAPE


def CAPE_integration(delta_CAPE):    # called by other functions only
    CAPE_val, buffer = 0, 0
    # If increment is +ve, increment the buffer;
    # if -ve, push (and reset) the buffer, the largest *pushed* buffer is CAPE
    # boundary case: if no pushed buffer CAPE=0; if delta(top of atmo) >0, it is not pushed
    for delta in delta_CAPE:
        if delta >= 0:
            buffer += delta
        elif buffer > CAPE_val:
            CAPE_val = buffer
            buffer = 0

    return CAPE_val


def CAPE(T_z_DS, plot=False, figsize=(20, 10)):
    T_ap, T_env, delta_z, z_profile = CAPE_param(T_z_DS)
    delta_CAPE = CAPE_discretize(T_ap, T_env, delta_z)
    CAPE_val = CAPE_integration(delta_CAPE)

    if plot:
        plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        ax = plot_T_z(T_ap, T_env, z_profile)
        plt.subplot(1, 2, 2)
        ax = plot_delta_CAPE(delta_CAPE, z_profile)
        plt.show()

    return CAPE_val


def list2da(data, data_name, dim, dim_name):
    return xr.DataArray(data,
                        coords={dim_name: dim},
                        dims=dim_name,
                        name=data_name)


def plot_T_z(T_ap, T_env, z_profile):
    ax = sns.lineplot(T_ap, z_profile)
    ax = sns.lineplot(T_env, z_profile)
    ax.set(xlabel='Temperature (K)', ylabel='Elevation (m)')
    return ax


def plot_delta_CAPE(delta_CAPE, z_profile):
    ax = sns.barplot(delta_CAPE, z_profile,
                     orient='h',
                     order=reversed(z_profile))
    ax.set(xlabel='CAPE increment', ylabel='Elevation (m)')
    return ax
