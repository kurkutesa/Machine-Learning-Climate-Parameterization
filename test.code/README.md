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
cd /exports/csce/datastore/geos/users/$USER/GitHub/Machine-Learning-Climate-Parameterization/main/stage-1/
gio open .
wget -m <ftp-link>
```

### Tensorboard
```
python -m tensorboard.main --logdir=./log/<runID>
localhost:6006
```

## PUMA
```
ssh -Y <UUN>@puma.nerc.ac.uk
umui &
rosie go &
ssh -l <k...> login.archer.ac.uk
qstat -u <k...>

grep -r <search> .
```

```
ps -flu <UUN> | grep <SUITE>
kill -9 <PID>
```

## Eddie
```
ssh -Y eddie
ssh <UUN>@eddie3.ecdf.ed.ac.uk
qsub
qstat
rsync
```

## Fix Rosie Go on Eddie
Add the following line at the bottom of ~/.bashrc
```
export PATH="$PATH:/exports/csce/eddie/geos/groups/cesd/sw/rose/bin"
```

## Git Basics
### Creation
```
git init <new_repo>
git clone https://github.com/edenau/<repo.git>
```
### Remote Link
```
git remote -v
git remote add upstream <upstream-repo.git>
git fetch upstream
git merge upstream/gh-pages
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
git reset <file>
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
ncdump -h
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

### .ipynb to .py
```
ipython nbconvert <file.ipynb> --to python
```

## Run Pascal in Mac OS
```
MACOSX_DEPLOYMENT_TARGET=10.12 fpc <hello.pas>
```
