# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : get_start_date.py
# POURPOSE : get start dates. Aproximates to the nearest multiple of 10
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 24/01/2019 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# System
import os
import sys
import datetime
import subprocess

# Dates
import pytz
from pytz import timezone

# Data
import xarray as xr
import pandas as pd

import warnings
warnings.filterwarnings("ignore")



def roundTime(dt=None, dateDelta=datetime.timedelta(minutes=1)):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
            Stijn Nevens 2014 - Changed to use only datetime objects as variables
    """
    roundTo = dateDelta.total_seconds()

    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)



if __name__ == '__main__':
    
    EST = timezone('Australia/Sydney')
    UTC = timezone("UTC")

    ds = xr.open_dataset(sys.argv[1])

    start = pd.to_datetime(ds.time.values[0]).to_pydatetime()

    # round to 10
    rounded = roundTime(dt=start, dateDelta=datetime.timedelta(minutes=10))

    if rounded < start:
        print("Problem!")

    now = UTC.localize(rounded)

    est = now.astimezone(EST)

    print(est.strftime("%Y%m%d-%H:%M:%S"))

