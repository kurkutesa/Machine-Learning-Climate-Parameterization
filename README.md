# Using Machine Learning to Parameterize Climate and Weather Processes: A Case-Study on Convection

This is an on-going project with the aim to parameterize subgrid climate processes that are too complex or small-scale to be explicitly represented and numerically computed in existing global climate models (GCMs), with the focus on emulating convection parameterization at the moment. 

## Directories
* [UM.teaching/](./UM.teaching/) contains python codes for teaching purpose. It demonstrates basic *xarray* operations to analyze data with netCDF format.

* [test.code/](./test.code/) contains python codes to test if all required python packages are intalled in your own environment.

* [work/](./work/) contains files related to the project.

* [work/NN/](./work/NN/) contains all machine learning-related codes using *TensorFlow* package.

* [work/NN/colab/](./work/NN/colab/) contains those machine learning-related codes in .ipynb format, optimzied for [Google Colaboratory](https://colab.research.google.com/) environment. They have same functionalities with .py equivalents, with minor code difference taking into account different environments. I am building codes to enable automatic conversion between .py and .ipynb files, as I leveraged both envionments for various tasks.

* Note that all raw data in NetCDF format are not available in the repo as they all exceed GitHub file size limit (100 MB).

## Objectives

Traditional atmospheric models used to forecast the weather a few days ahead or to project climate change over the next century require parameterizations for physical processes that cannot be directly represented by the resolved large-scale flow. Parameterization corresponds to approximating effects of the unresolved processes as a function of the resolved flow. Uncertainties in climate projections for the next 20 to 50 years largely arise from uncertainty in how unresolved processes are parameterized, for which convection is one of the key processes. Convection arises because the atmosphere is heated from below and air becomes unstable due to vertical density gradient. At which point it mixes vertically, generating deep cloud layers, heavy rain, and atmospheric heating which then drive the atmospheric circulation. This on-going project aims to take advantage of rich datasets of observations from *in situ* measurements and remote sensing to construct a function using machine learning (ML) that parameterizes convection as a case study to parameterize many processes in atmospheric models.

Current approaches to parameterization use theories to generate functions which are then embedded within climate models. The parameterization acts on the resolved state and produces rates of change which along with other processes are used to update the model state. ML aims to generate a function that empirically relies on data. There is a rich array of atmospheric data available to train, validate, and test ML algorithms, and many possible algorithms that could be used. One challenge for ML algorithms is to conserve, amongst others, total energy, moisture and momentum, in other words to be physics-aware.

The proposed approach is to first use Single Column Model (SCM) configuration of the Met Office weather and forecasting model, and multiple years of *in situ* data from three Atmospheric Radiation Measurement (ARM) climate research facility sites. The SCM contains all the parameterizations used in the full atmospheric model but needs to be driven by the large-scale dynamical state taken from 're-analysis', for example the ERA-Interim data from European Centre for Medium-Range Weather Forecasts (ECMWF). Then the SCM needs to be modified to remove the current convection parameterization and add an interface to existing ML algorithms. In the ARM dataset, days on which convection is the dominant process will be identified by quantitative metrics. This data subset will then be split into training and test datasets for training and evaluating the ML model (or ensemble of models). Results of the model will be compared with the reference SCM to check if ML is able to emulate traditional parameterizations.

The final stage is to examine what benefits there are from using the enormous amount of remote sensed data over the entire globe. Rather than running the SCM at a small number of points but for many years, the approach is to run the SCM at all points across the Earth but for only one year. To test the trained ML algorithm, another year will be selected and, as above, the standard SCM and modified SCM will be run and their errors compared.
