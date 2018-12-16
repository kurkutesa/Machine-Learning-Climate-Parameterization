# Command Cheat Sheet

## Burn Server
### Virtual Environment
```
export PATH="/scratch/$USER/miniconda/base/bin:$PATH"
source activate venv

source deactivate
```
### DataStore Directory
```
cd /exports/csce/datastore/geos/users/$USER/GitHub/Machine-Learning-Climate-Parameterization/work/NN/
```

### Tensorboard
```
python -m tensorboard.main --logdir=./log/<runID>
localhost:6006
```

### .ipynb to .py
```
ipython nbconvert <file.ipynb> --to python
```

## Eddie
```
qsub
qstat
rsync
```

## Git Basics
### Creation
```
git init
git clone https://github.com/edenau/<repo.git>
```
### Pull
```
git fetch
git pull
git pull origin <branch>
```
### Push
```
git status
git add -A
git commit -m "MESSAGE"
git push -u origin master
```
### Miscellaneous
```
git remote set-url <url>
git log
```

## Python
### *xarray* Basics
```
ncdump
ncview

import xarray as xr
DS = xr.open_dataset('data.cdf')
DS = xr.open_mfdataset('*.cdf')
DS.var
DS.coords['lat'].values
da.sel(lat=50, method='nearest')
da.time.dt.year/dayofweek/dayofyear/month/day.to_index()
DS_da.merge([T,rh,u,v])
DS.to_dataframe
```

## Run Pascal in Mac OS
```
MACOSX_DEPLOYMENT_TARGET=10.12 fpc <hello.pas>
```
