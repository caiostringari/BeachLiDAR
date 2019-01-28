# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : LiDAR2nc.py
# POURPOSE : Convert LSM5X exported CSV files to netcdf
#
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 21/03/2018 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import os

import sys

import subprocess

import argparse

import pytz
import datetime

import xarray as xr

import math
import numpy as np

import pandas as pd

from scipy.spatial.distance import cdist

# Progress bar
from tqdm import tqdm

import string
import random


# Plotting
import seaborn as sns
import matplotlib.pyplot as plt


def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for m in range(length))


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def parsetime(strtime):
    times = []
    for t in strtime:
        now = datetime.datetime.strptime(t, DATE_FMT)
        now_tz = now.replace(tzinfo=pytz.timezone(TIMEZONE))
        # to utc
        utctime = now_tz.astimezone(pytz.utc)
        times.append(utctime)
    return times


def merger(files, ncout='merged.nc'):
    """
    ----------
    Args:
        files [Mandatory (list, np.ndarray)]: Sorted list of files to merge.

        ncout [Optional (str)]: Output filename. Defaul is timestack.nc

        ramcut [Optinal (float)]: Max fraction of memory to use. If the size of
        the expected merged file exceds this fraction will raise a MemoryError.

    ----------
    Returns:
    """
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # load dataset into memory
            ds.load()
            return ds

    # loop over files
    datasets = []
    with tqdm(total=len(files)) as pbar:
        for k, fname in enumerate(files):
            datasets.append(process_one_path(fname))
            pbar.update()
    pbar.close()
    merged = xr.concat(datasets, "time")
    merged.to_netcdf(ncout)
    merged.close()


def main():

    print("\nProcessing raw data, please wait...\n")

    # IO names
    input_file = args.input[0]
    output_file = args.output[0]

    # total number of lines
    nlines = sum(1 for row in open(input_file, 'r'))
    N = int(nlines/CHUNKSIZE)

    # temporaty folder
    tmpfolder = random_string(10)
    os.mkdir(tmpfolder)

    # loop over file chunks
    k = 0
    files = []
    reader = pd.read_csv(input_file, chunksize=CHUNKSIZE)
    with tqdm(total=N) as pbar:
        for df in reader:

            # drop all the useless columns
            df.dropna(axis=1, inplace=True)

            # get time
            strtime = df["Recording time"].values

            # parse time values
            time = parsetime(strtime)

            # read chanell 16 data
            df_channel16 = df.filter(regex="ScanData.aDataChannel16")

            # data in milimeters
            df_data = df_channel16.filter(regex=r"aData\[")

            # convert to meters
            data = df_data.values*SCALE_FACTOR

            # angles
            angle_range = np.deg2rad(
                np.linspace(-5, 185, len(data[0, :])))

            # to cartesian coords
            x, y = pol2cart(data, angle_range)

            # build the dataset
            ds = xr.Dataset()
            # coordinates
            ds.coords['time'] = time
            ds.coords["points"] = np.arange(0, len(angle_range), 1)
            # data values
            ds['x'] = (('time', 'points'),  x)  # x-coodinates
            ds['y'] = (('time', 'points'),  y)  # y-coodinates
            ds["distance"] = (('time', 'points'),  data)  # distance
            ds["angles"] = (('points'), np.rad2deg(angle_range))

            # time encoding
            units = 'days since 2000-01-01 00:00:00'
            calendar = 'gregorian'
            encoding = dict(time=dict(units=units, calendar=calendar))

            # temporary file name
            fname = "{}/{}.nc".format(tmpfolder, str(k).zfill(4))
            files.append(fname)

            # dump to file
            ds.to_netcdf(fname, encoding=encoding)

            k += 1
            pbar.update()
    pbar.close()

    # merge everything
    print("\nMerging file, please wait...\n")
    merger(files, output_file)

    # clean
    subprocess.call("rm -rf {}".format(tmpfolder), shell=True)

    print("\nMy work is done!\n")


if __name__ == '__main__':
    try:
        inp = sys.argv[1]
    except Exception:
        raise IOError(
            "Usage: LiDAR2nc input.csv output.nc")
    if inp in ["-i", "--input"]:
        # argument parser
        parser = argparse.ArgumentParser()
        # input data
        parser.add_argument('--input', '-i',
                            nargs=1,
                            action='store',
                            dest='input',
                            help="Input CSV file.",
                            required=True)
        parser.add_argument('--output', '-o',
                            nargs=1,
                            action='store',
                            dest='output',
                            help="Output netCDF file.",
                            required=True)
        args = parser.parse_args()
    else:
        if len(sys.argv) < 3:
            raise IOError(
                "Usage: LiDAR2nc input.csv output.nc")
        # argument parser
        parser = argparse.ArgumentParser()
        # input data
        parser.add_argument('--input', '-i',
                            nargs=1,
                            action='store',
                            dest='input',
                            help="Input CSV file.",)
        parser.add_argument('--output', '-o',
                            nargs=1,
                            action='store',
                            dest='output',
                            help="Output netCDF file.",
                            required=True)
        # parser
        args = parser.parse_args(["-i", sys.argv[1], "-o", sys.argv[2]])
    # main()

    # constants
    CHUNKSIZE = 10 ** 3
    SCALE_FACTOR = 0.001
    TIMEZONE = "Australia/Sydney"
    DATE_FMT = "%Y-%m-%d %H:%M:%S.%f AEST(+1000)"

    # main
    main()
