#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

server = ECMWFDataServer()

server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2000-01-01/to/2009-12-31",
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
    "target": "era_interim_vv_500hpa.nc",
})
