#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer

# May need to do this in command line
#pip install --user https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-api-client-python.tgz

server = ECMWFDataServer()

'''
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
#130.128/131.128/132.128/135.128/157.128#
'''

server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "2013-01-01/to/2015-01-01",
    "expver": "1",
    "grid": "0.125/0.125",
    "area": "-1.5/147/-2.5/148", # Around ARM PNG site
    #"levelist": "1/2/3/5/7/10/20/30/50/70/100/125/150/175/200/225/250/300/350/400/450/500/550/600/650/700/750/775/800/825/850/875/900/925/950/975/1000",
    "levtype": "sfc",
    #"levtype": "pl",
    #"param": "130.128/131.128/132.128/135.128/157.128", # parameters with p levels
    "param": "50.128/142.128/143.128/228.128",
    #"step": "0",
    "step": "3/6/9/12",
    "stream": "oper",
    #"time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "time": "00:00:00/12:00:00",
    #"type": "an",
    "type": "fc",
    "format":"netcdf",
    "target": "../forNN/erainterim.PNG.201314.nc",
})
