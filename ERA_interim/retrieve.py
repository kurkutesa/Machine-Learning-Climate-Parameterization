#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

# May need to do this in command line
#pip install --user https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-api-client-python.tgz

server = ECMWFDataServer()

server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "1990-01-01/to/1999-12-31",
    "expver": "1",
    "grid": "0.75/0.75",
    "levelist": "500",
    "levtype": "pl",
    "param": "135.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    "format":"netcdf",
    "target": "data/era_interim_vv_500hpa_199x.nc",
})
