""" Functions for processing data from the ximu IMU sensors. """

__version__ = '0.2'
__author__ = 'Kjartan Halvorsen'

import sys
import os
import numpy as np
import math
import csv
import itertools
import warnings
import sqlite3
import h5py
import unittest
import functools
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

from scipy.interpolate import interp1d
import scipy.optimize as optimize
from scipy.signal import detrend, bessel, filtfilt
import scipy.io as sio
import scipy.stats

from datetime import datetime, timedelta, date

import logging
logging.basicConfig(filename='ximudata.log',level=logging.DEBUG)
import xlsxwriter

from nvg.maths import quaternions as quat
from nvg.algorithms import orientation
#from nvg.utilities import time_series
from nvg.ximu import pointfinder
from nvg.ximu import markerdata
from nvg.ximu import kinematics
from nvg.io import qualisys_tsv as qtsv
from nvg.ximu import markerdata

from cyclicpython import cyclic_path
from cyclicpython.algorithms import kinematics as cpkinematics
#from cyclicpython.algorithms import fomatlab as fomatlab
from cyclicpython.algorithms import ekf as cpekf
from cyclicpython.algorithms import detect_peaks
from cyclicpython import cyclic_planar as cppl

def nvg_2012_09_data(dtaroot = "/media/ubuntu-15-10/home/kjartan/nvg/"):
    """
        Lists all data files for the nvg project acquired in September 2012.
        OBS: The eventlogs are missing.

        New 2017-09-15: Marker data available with key "MD-N" and "MD-D"
    """
    dp = dtaroot + "2012-09-17/S1/"
    s1 = {}
    s1["LA"] = [dp + "LA-200/NVG_2012_S1_A_LA_00201_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S1_A_LA_00201_CalInertialAndMag.csv"]
    s1["LH"] = [dp + "LH-800/NVG_2012_S1_A_LH_00801_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S1_A_LH_00801_CalInertialAndMag.csv"]
    s1["LT"] = [dp + "LT-400/NVG_2012_S1_A_LT_00401_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S1_A_LT_00401_CalInertialAndMag.csv"]
    s1["N"] = [dp + "N-600/NVG_2012_S1_A_N_00601_DateTime.csv", \
                    dp + "N-600/NVG_2012_S1_A_N_00601_CalInertialAndMag.csv"]
    s1["B"] = []
    s1["RA"] = [dp + "RA-100/NVG_2012_S1_A_RA_00101_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S1_A_RA_00101_CalInertialAndMag.csv"]
    s1["RH"] = [dp + "RH-700/NVG_2012_S1_A_RH_00701_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S1_A_RH_00701_CalInertialAndMag.csv"]
    s1["RT"] = [dp + "RT-300/NVG_2012_S1_A_RT_00301_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S1_A_RT_00301_CalInertialAndMag.csv"]

    s1events = dp + "NVG_2012_S1_eventlog_fix"

    dp = dtaroot + "2012-09-18/S2/"
    s2 = {}
    s2["LA"] = [dp + "LA-200/NVG_2012_S2_A_LA_00201_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S2_A_LA_00201_CalInertialAndMag.csv"]
    s2["LH"] = [dp + "LH-800/NVG_2012_S2_A_LH_00801_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S2_A_LH_00801_CalInertialAndMag.csv"]
    s2["LT"] = [dp + "LT-400/NVG_2012_S2_A_LT_00401_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S2_A_LT_00401_CalInertialAndMag.csv"]
    s2["N"] = [dp + "N-600/NVG_2012_S2_A_N_00601_DateTime.csv", \
                    dp + "N-600/NVG_2012_S2_A_N_00601_CalInertialAndMag.csv"]
    s2["B"] = []
    s2["RA"] = [dp + "RA-100/NVG_2012_S2_A_RA_00101_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S2_A_RA_00101_CalInertialAndMag.csv"]
    s2["RH"] = [dp + "RH-700/NVG_2012_S2_A_RH_00701_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S2_A_RH_00701_CalInertialAndMag.csv"]
    s2["RT"] = [dp + "RT-300/NVG_2012_S2_A_RT_00301_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S2_A_RT_00301_CalInertialAndMag.csv"]

    s2events = dp + "NVG_2012_S2_eventlog_fix"

    dp = dtaroot + "2012-09-18/S3/"
    s3 = {}
    s3["LA"] = [dp + "LA-200/NVG_2012_S3_A_LA_00202_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S3_A_LA_00202_CalInertialAndMag.csv"]
    s3["LH"] = [dp + "LH-800/NVG_2012_S3_A_LH_00802_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S3_A_LH_00802_CalInertialAndMag.csv"]
    s3["LT"] = [dp + "LT-400/NVG_2012_S3_A_LT_00402_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S3_A_LT_00402_CalInertialAndMag.csv"]
    s3["N"] = [dp + "N-600/NVG_2012_S3_A_N_00602_DateTime.csv", \
                    dp + "N-600/NVG_2012_S3_A_N_00602_CalInertialAndMag.csv"]
    s3["B"] = []
    s3["RA"] = [dp + "RA-100/NVG_2012_S3_A_RA_00102_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S3_A_RA_00102_CalInertialAndMag.csv"]
    s3["RH"] = [dp + "RH-700/NVG_2012_S3_A_RH_00702_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S3_A_RH_00702_CalInertialAndMag.csv"]
    s3["RT"] = [dp + "RT-300/NVG_2012_S3_A_RT_00302_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S3_A_RT_00302_CalInertialAndMag.csv"]

    s3events = dp + "NVG_2012_S3_eventlog"

    dp = dtaroot + "2012-09-19-S4/S4/"
    s4 = {}
    s4["MD-N"] = dtaroot + "/Data/S4/NVG_2012_S4_N.tsv"
    s4["MD-D"] = dtaroot + "/Data/S4/NVG_2012_S4_D.tsv"
    s4["LA"] = [dp + "LA-200/NVG_2012_S4_A_LA_00203_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S4_A_LA_00203_CalInertialAndMag.csv"]
    s4["LH"] = [dp + "LH-800/NVG_2012_S4_A_LH_00803_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S4_A_LH_00803_CalInertialAndMag.csv"]
    s4["LT"] = [dp + "LT-400/NVG_2012_S4_A_LT_00403_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S4_A_LT_00403_CalInertialAndMag.csv"]
    s4["N"] = [dp + "N-600/NVG_2012_S4_A_N_00603_DateTime.csv", \
                    dp + "N-600/NVG_2012_S4_A_N_00603_CalInertialAndMag.csv"]
    s4["B"] = [dp + "B-500/NVG_2012_S4_A_B_00503_DateTime.csv", \
                    dp + "B-500/NVG_2012_S4_A_B_00503_CalInertialAndMag.csv"]
    s4["RA"] = [dp + "RA-100/NVG_2012_S4_A_RA_00103_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S4_A_RA_00103_CalInertialAndMag.csv"]
    s4["RH"] = [dp + "RH-700/NVG_2012_S4_A_RH_00703_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S4_A_RH_00703_CalInertialAndMag.csv"]
    s4["RT"] = [dp + "RT-300/NVG_2012_S4_A_RT_00303_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S4_A_RT_00303_CalInertialAndMag.csv"]

    s4events = dp + "NVG_2012_S4_eventlog"


    dp = dtaroot + "2012-09-19-S5/S5/"
    s5 = {}
    s5["LA"] = [dp + "LA-200/NVG_2012_S5_A_LA_00204_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S5_A_LA_00204_CalInertialAndMag.csv"]
    s5["LH"] = [dp + "LH-800/NVG_2012_S5_A_LH_00804_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S5_A_LH_00804_CalInertialAndMag.csv"]
    s5["LT"] = [dp + "LT-400/NVG_2012_S5_A_LT_00404_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S5_A_LT_00404_CalInertialAndMag.csv"]
    s5["N"] = [dp + "N-600/NVG_2012_S5_A_N_00604_DateTime.csv", \
                    dp + "N-600/NVG_2012_S5_A_N_00604_CalInertialAndMag.csv"]
    s5["B"] = [dp + "B-500/NVG_2012_S5_A_B_00504_DateTime.csv", \
                    dp + "B-500/NVG_2012_S5_A_B_00504_CalInertialAndMag.csv"]
    s5["RA"] = [dp + "RA-100/NVG_2012_S5_A_RA_00104_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S5_A_RA_00104_CalInertialAndMag.csv"]
    s5["RH"] = [dp + "RH-700/NVG_2012_S5_A_RH_00704_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S5_A_RH_00704_CalInertialAndMag.csv"]
    s5["RT"] = [dp + "RT-300/NVG_2012_S5_A_RT_00304_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S5_A_RT_00304_CalInertialAndMag.csv"]

    s5events = dp + "NVG_2012_S5_eventlog"


    dp = dtaroot + "2012-09-19-S6/S6/"
    s6 = {}
    s6["MD-N"] = dtaroot + "/Data/S6/NVG_2012_S6_N.tsv"
    s6["MD-D"] = dtaroot + "/Data/S6/NVG_2012_S6_D.tsv"
    s6["LA"] = [dp + "LA-200/NVG_2012_S6_A_LA_00205_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S6_A_LA_00205_CalInertialAndMag.csv"]
    s6["LH"] = [dp + "LH-800/NVG_2012_S6_A_LH_00805_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S6_A_LH_00805_CalInertialAndMag.csv"]
    s6["LT"] = [dp + "LT-400/NVG_2012_S6_A_LT_00405_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S6_A_LT_00405_CalInertialAndMag.csv"]
    s6["N"] = [dp + "N-600/NVG_2012_S6_A_N_00605_DateTime.csv", \
                    dp + "N-600/NVG_2012_S6_A_N_00605_CalInertialAndMag.csv"]
    s6["B"] = [dp + "B-500/NVG_2012_S6_A_B_00505_DateTime.csv", \
                   dp + "B-500/NVG_2012_S6_A_B_00505_CalInertialAndMag.csv"]
    s6["RA"] = [dp + "RA-100/NVG_2012_S6_A_RA_00105_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S6_A_RA_00105_CalInertialAndMag.csv"]
    s6["RH"] = [dp + "RH-700/NVG_2012_S6_A_RH_00705_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S6_A_RH_00705_CalInertialAndMag.csv"]
    s6["RT"] = [dp + "RT-300/NVG_2012_S6_A_RT_00305_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S6_A_RT_00305_CalInertialAndMag.csv"]

    s6events = dp + "NVG_2012_S6_eventlog"


    dp = dtaroot + "2012-09-20/S7/"
    s7 = {}
    s7["LA"] = [dp + "NVG_2012_S7_A_LA_00206_DateTime.csv", \
                    dp + "NVG_2012_S7_A_LA_00206_CalInertialAndMag.csv"]
    s7["LH"] = [dp + "NVG_2012_S7_A_LH_00806_DateTime.csv", \
                    dp + "NVG_2012_S7_A_LH_00806_CalInertialAndMag.csv"]
    s7["LT"] = [dp + "NVG_2012_S7_A_LT_00406_DateTime.csv", \
                    dp + "NVG_2012_S7_A_LT_00406_CalInertialAndMag.csv"]
    s7["N"] = [dp + "NVG_2012_S7_A_N_00606_DateTime.csv", \
                    dp + "NVG_2012_S7_A_N_00606_CalInertialAndMag.csv"]
    s7["B"] = [dp + "NVG_2012_S7_A_B_00506_DateTime.csv", \
                    dp + "NVG_2012_S7_A_B_00506_CalInertialAndMag.csv"]
    s7["RA"] = [dp + "NVG_2012_S7_A_RA_00106_DateTime.csv", \
                    dp + "NVG_2012_S7_A_RA_00106_CalInertialAndMag.csv"]
    s7["RH"] = [dp + "NVG_2012_S7_A_RH_00706_DateTime.csv", \
                    dp + "NVG_2012_S7_A_RH_00706_CalInertialAndMag.csv"]
    s7["RT"] = [dp + "NVG_2012_S7_A_RT_00306_DateTime.csv", \
                    dp + "NVG_2012_S7_A_RT_00306_CalInertialAndMag.csv"]

    s7events = dp + "NVG_2012_S7_eventlog"


    dp = dtaroot + "2012-09-21/S8/"
    s8 = {}
    s8["LA"] = [dp + "LA-200/NVG_2012_S8_A_LA_00207_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S8_A_LA_00207_CalInertialAndMag.csv"]
    s8["LH"] = [dp + "LH-800/NVG_2012_S8_A_LH_00807_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S8_A_LH_00807_CalInertialAndMag.csv"]
    s8["LT"] = [dp + "LT-400/NVG_2012_S8_A_LT_00407_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S8_A_LT_00407_CalInertialAndMag.csv"]
    s8["N"] = [dp + "N-600/NVG_2012_S8_A_N_00607_DateTime.csv", \
                    dp + "N-600/NVG_2012_S8_A_N_00607_CalInertialAndMag.csv"]
    s8["B"] = [dp + "B-500/NVG_2012_S8_A_B_00507_DateTime.csv", \
                    dp + "B-500/NVG_2012_S8_A_B_00507_CalInertialAndMag.csv"]
    s8["RA"] = [dp + "RA-100/NVG_2012_S8_A_RA_00107_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S8_A_RA_00107_CalInertialAndMag.csv"]
    s8["RH"] = [dp + "RH-700/NVG_2012_S8_A_RH_00707_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S8_A_RH_00707_CalInertialAndMag.csv"]
    s8["RT"] = [dp + "RT-300/NVG_2012_S8_A_RT_00307_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S8_A_RT_00307_CalInertialAndMag.csv"]

    s8events = dp + "NVG_2012_S8_eventlog"


    dp = dtaroot + "2012-09-21/S9/"
    s9 = {}
    s9["LA"] = [dp + "LA-200/NVG_2012_S9_A_LA_00208_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S9_A_LA_00208_CalInertialAndMag.csv"]
    s9["LH"] = [dp + "LH-800/NVG_2012_S9_A_LH_00808_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S9_A_LH_00808_CalInertialAndMag.csv"]
    s9["LT"] = [dp + "LT-400/NVG_2012_S9_A_LT_00408_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S9_A_LT_00408_CalInertialAndMag.csv"]
    s9["N"] = [dp + "N-600/NVG_2012_S9_A_N_00608_DateTime.csv", \
                    dp + "N-600/NVG_2012_S9_A_N_00608_CalInertialAndMag.csv"]
    s9["B"] = [dp + "B-500/NVG_2012_S9_A_B_00508_DateTime.csv", \
                    dp + "B-500/NVG_2012_S9_A_B_00508_CalInertialAndMag.csv"]
    s9["RA"] = [dp + "RA-100/NVG_2012_S9_A_RA_00108_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S9_A_RA_00108_CalInertialAndMag.csv"]
    s9["RH"] = [dp + "RH-700/NVG_2012_S9_A_RH_00708_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S9_A_RH_00708_CalInertialAndMag.csv"]
    s9["RT"] = [dp + "RT-300/NVG_2012_S9_A_RT_00308_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S9_A_RT_00308_CalInertialAndMag.csv"]

    s9events = dp + "NVG_2012_S9_eventlog"


    dp = dtaroot + "2012-09-24-S10/S10/"
    s10 = {}
    s10["MD-N"] = dtaroot + "/Data/S10/NVG_2012_S10_N.tsv"
    s10["MD-D"] = dtaroot + "/Data/S10/NVG_2012_S10_D.tsv"

    s10["LA"] = [dp + "LA-200/NVG_2012_S10_A_LA_00209_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S10_A_LA_00209_CalInertialAndMag.csv"]
    s10["LH"] = [dp + "LH-800/NVG_2012_S10_A_LH_00809_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S10_A_LH_00809_CalInertialAndMag.csv"]
    s10["LT"] = [dp + "LT-400/NVG_2012_S10_A_LT_00409_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S10_A_LT_00409_CalInertialAndMag.csv"]
    s10["N"] = [dp + "N-600/NVG_2012_S10_A_N_00609_DateTime.csv", \
                    dp + "N-600/NVG_2012_S10_A_N_00609_CalInertialAndMag.csv"]
    s10["B"] = [dp + "B-500/NVG_2012_S10_A_B_00509_DateTime.csv", \
                    dp + "B-500/NVG_2012_S10_A_B_00509_CalInertialAndMag.csv"]
    s10["RA"] = [dp + "RA-100/NVG_2012_S10_A_RA_00109_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S10_A_RA_00109_CalInertialAndMag.csv"]
    s10["RH"] = [dp + "RH-700/NVG_2012_S10_A_RH_00709_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S10_A_RH_00709_CalInertialAndMag.csv"]
    s10["RT"] = [dp + "RT-300/NVG_2012_S10_A_RT_00309_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S10_A_RT_00309_CalInertialAndMag.csv"]

    s10events = dp + "NVG_2012_S10_eventlog"


    dp = dtaroot + "2012-09-24-S11/S11/"
    s11 = {}
    s11["LA"] = [dp + "LA-200/NVG_2012_S11_A_LA_00210_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S11_A_LA_00210_CalInertialAndMag.csv"]
    s11["LH"] = [dp + "LH-800/NVG_2012_S11_A_LH_00810_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S11_A_LH_00810_CalInertialAndMag.csv"]
    s11["LT"] = [dp + "LT-400/NVG_2012_S11_A_LT_00410_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S11_A_LT_00410_CalInertialAndMag.csv"]
    s11["N"] = [dp + "N-600/NVG_2012_S11_A_N_00610_DateTime.csv", \
                    dp + "N-600/NVG_2012_S11_A_N_00610_CalInertialAndMag.csv"]
    s11["B"] = [dp + "B-500/NVG_2012_S11_A_B_00510_DateTime.csv", \
                    dp + "B-500/NVG_2012_S11_A_B_00510_CalInertialAndMag.csv"]
    s11["RA"] = [dp + "RA-100/NVG_2012_S11_A_RA_00110_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S11_A_RA_00110_CalInertialAndMag.csv"]
    s11["RH"] = [dp + "RH-700/NVG_2012_S11_A_RH_00710_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S11_A_RH_00710_CalInertialAndMag.csv"]
    s11["RT"] = [dp + "RT-300/NVG_2012_S11_A_RT_00310_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S11_A_RT_00310_CalInertialAndMag.csv"]

    s11events = dp + "NVG_2012_S11_eventlog"


    dp = dtaroot + "2012-09-24-S12/S12/"
    s12 = {}
    s12["MD-N"] = dtaroot + "/Data/S12/NVG_2012_S12_N.tsv"
    s12["MD-D"] = dtaroot + "/Data/S12/NVG_2012_S12_D.tsv"
    s12["LA"] = [dp + "LA-200/NVG_2012_S12_A_LA_00211_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S12_A_LA_00211_CalInertialAndMag.csv"]
    s12["LH"] = [dp + "LH-800/NVG_2012_S12_A_LH_00811_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S12_A_LH_00811_CalInertialAndMag.csv"]
    s12["LT"] = [dp + "LT-400/NVG_2012_S12_A_LT_00411_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S12_A_LT_00411_CalInertialAndMag.csv"]
    s12["N"] = [dp + "N-600/NVG_2012_S12_A_N_00611_DateTime.csv", \
                    dp + "N-600/NVG_2012_S12_A_N_00611_CalInertialAndMag.csv"]
    s12["B"] = [dp + "B-500/NVG_2012_S12_A_B_00511_DateTime.csv", \
                    dp + "B-500/NVG_2012_S12_A_B_00511_CalInertialAndMag.csv"]
    s12["RA"] = [dp + "RA-100/NVG_2012_S12_A_RA_00111_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S12_A_RA_00111_CalInertialAndMag.csv"]
    s12["RH"] = [dp + "RH-700/NVG_2012_S12_A_RH_00711_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S12_A_RH_00711_CalInertialAndMag.csv"]
    s12["RT"] = [dp + "RT-300/NVG_2012_S12_A_RT_00311_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S12_A_RT_00311_CalInertialAndMag.csv"]

    s12events = dp + "NVG_2012_S12_eventlog"


    return [dict(S2=s2, S3=s3, S4=s4, S5=s5, S6=s6, S7=s7, S8=s8, S9=s9, \
                     S10=s10, S11=s11, S12=s12), \
                dict(S2=s2events, S3=s3events, S4=s4events, S5=s5events, S6=s6events,\
                         S7=s7events, S8=s8events, S9=s9events, S10=s10events, \
                         S11=s11events, S12=s12events)]


# Global since this mapping should never be changed
IMU_MARKERS = {}
IMU_MARKERS["LT"] = dict(upper="HIP", lower="KNEE",  closest="THIGH")
IMU_MARKERS["LA"] = dict(upper="KNEE", lower="ANKLE",  closest="ANKLE")
IMU_MARKERS["N"] = dict(upper="C7", lower="SACRUM",  closest="C7")
IMU_MARKERS["LH"] = dict(upper="ELBOW", lower="WRIST",  closest="WRIST")
IMU_MARKERS["B"] = dict(upper="C7", lower="SACRUM",  closest="SACRUM")



class NVGData:

    def __init__(self, hdfFilename, mode='r', debug=False):
        """
        Opens database.

        Arguments
        hdfFilename   -> Name of database file
        mode          -> Mode for opening database. Default is read-only
        debug         -> If true logs and plots
        """
        self.fname = hdfFilename
        # No need: self.create_nvg_db(). Better to just add subject data
        self.hdfFile = h5py.File(self.fname, mode)

        self.debug = debug

        self.rotationEstimator = kinematics.CyclicEstimator(14) # Default is cyclic estimator
        # Use tracking algorithm from nvg. Restart at each cycle start
        #self.rotationEstimator = kinematics.GyroIntegratorOrientation(self.hdfFile.attrs['packetNumbersPerSecond'])

        self.displacementEstimator = kinematics.IntegrateAccelerationDisplacementEstimator()

        #self.displacementEstimator = self.CyclicPlanarDisplacementEstimator(14)


    def close(self):
        self.hdfFile.close()

    def create_nvg_db(self):
        """
        Creates the hd5 database for the nvg project and adds a subgroup for each subject.
        """
        try:
            f = h5py.File(self.fname)
            for s in range(2,13):
                sstr = "S%d"% (s,)
                g = None
                try:
                    g = f.create_group(sstr)
                except ValueError:
                    logging.info("Sub group " + sstr + " already exists")
                    g = f["/"+sstr]
                    for c in ['N', 'D', 'B', 'M']:
                        try:
                            g.create_group(c)
                        except ValueError:
                            logging.info("Sub group " + c + " already exists")

            self.hdfFile = f
        except ValueError:
            logging.warning("Error opening file " + fname)

    def list_imus(self):
        """ Makes a list of all imus for which data exists for each trial """
        #dlist = defaultdict(list)
        def checkd(name, obj):
            if name[-1] in ["N", "B", "M", "D"]: # is trial or B imu
                if not obj.parent.name[-1] in ["N", "B", "M", "D"]: # obj is trial
                    print name, obj.keys()
        self.hdfFile.visititems(checkd)
        #return dlist

    def apply_to_all_trials(self, func, kwparams=None, subjlist=[], triallist=[]):
        """ Will call the function func for each subject and each trial in the data set.
        Whatever is returned from the function is collected in a dict indexed by
        the tuple (subject, trial).
        Example::
           nvgDB = ximudata.NVGData()
           res = nvgDB.apply_to_all_trials(nvgDB.get_RoM_angle_to_vertical, {'imu':'N'},
                                           subjlist=['S2', 'S3'])
        Which will apply the provided function, with the keyword argument imu='N' to all trials
        for subjects S2 and S3.
        """
        result = {}
        for subject in self.hdfFile:
            if subjlist == [] or subject in subjlist:
                print "-----------------------------------------"
                print "Applying function to subject %s" % (subject,)
                subj = self.hdfFile[subject]
                for trial in subj:
                    if triallist == [] or trial in triallist:
                        pyplot.close('all')
                        print "Applying to trial %s" % (trial,)
                        try:
                            if kwparams == None:
                                res = func(subject, trial)
                            else:
                                res = func(subject, trial, **kwparams)
                                result[(subject, trial)] = res
                        except:
                            print "Exception occurred! Check data. Proceeding..."
                print "-----------------------------------------"
        return result

    def get_trial_attribute(self, subject, trial, attrname):
        subj = self.hdfFile[subject]
        tr = subj[trial]
        if attrname in tr.attrs.keys():
            return tr.attrs[attrname]
        else:
            return None

    def descriptive_statistics(self, res):
        """
        Iterates over the dict res creating an returning a similar dict,
        but with descriptive statistics on the data elements in res.
        If any data element is empty, this particular element will be missing
        in the returned dict.
        """
        stats = {}
        for (key, result) in res.iteritems():
            if len(result) > 1:
                stats[key] = [np.mean(result), np.std(result), np.min(result),
                        np.percentile(result, 25), np.median(result),
                        np.percentile(result, 75), np.max(result)]

        return stats

    def descriptive_statistics_decorator(self, func):
        """
        Decorator that will call the provided function to get data,
        then calculate descriptive statistics and returning the
        list of  values [mean, std, min, Q1, Q2, Q3, max]
        typical usage::
           nvgDB = NVGData()
           nvgDB.apply_to_all_trials( \
              nvgDB.descriptive_statistics_decorator(nvgDB.get_trial_attribute),
              dict(attrname="cycleFrequency"))
        """
        def wrapper(self,*args,**kwargs):
            result = func(self,*args, **kwargs)
            return [np.mean(result), np.std(result), np.min(result),
                    np.percentile(result, 25), np.median(result),
                    np.percentile(result, 75), np.max(result)]
        return wrapper

    def scale_decorator(self, func, scale=1.0):
        """ Decorator that will call the provided function to get data,
        then scale the values.
        typical usage::
           nvgDB = NVGData()
           nvgDB.apply_to_all_trials(
              nvgDB.scale_decorator(nvgDB.get_rotation, scale=180.0/np.pi),
              dict(imu="B", startTime=120, anTime=60, doPlots=False)))
              attrname="cycleFrequency") )
        """

        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return [scale*np.asarray(r_) for r_ in result]
        return wrapper

    def range_decorator(self, func):
        """ Decorator that will call the provided function to get data,
        then calculate the range of the data
        typical usage::
           nvgDB = NVGData()
           nvgDB.apply_to_all_trials(
              nvgDB.range_decorator(nvgDB.get_vertical_displacement),
              dict(imu="B", startTime=120, anTime=60, doPlots=False)))
              attrname="cycleFrequency") )
        """

        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return [np.max(r_) - np.min(r_) for r_ in result]
        return wrapper

    def minmax_decorator(self, func, start=0.0, end=1.0):
        """ Decorator that will call the provided function to get data,
        then calculate the min and max of the data. The optional arguments
        start and end define start and end of an interval in the gait cycle
        for which to look for min and max.
        typical usage::
           nvgDB = NVGData()
           nvgDB.apply_to_all_trials(
              nvgDB.range_decorator(nvgDB.get_vertical_displacement),
              dict(imu="B", startTime=120, anTime=60, doPlots=False)))
              attrname="cycleFrequency") )
        """

        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            minmax = []
            for r_ in result:
                cycleLength = len(r_)
                startInd = int(start*cycleLength)
                endInd = int(end*cycleLength)-1
                minmax.append((np.min(r_[startInd:endInd]),
                        np.max(r_[startInd:endInd])))
            return [np.max(r_) - np.min(r_) for r_ in result]
        return wrapper

    def normalize_statistics(self, stats):
        """ Goes through the dict stats containing desriptive statistics,
        and normalize for each trial to the corresponding value for the Normal
        trial.
        """
        results = {}
        for ((subj, trial), vals) in stats.items():
            normals = stats[(subj, "N")]
            results[(subj,trial)] = [val_/norm_ for (val_, norm_) \
                                         in itertools.izip(vals,normals)]
        return results


    def test_difference_in_condition(self, results, title,
                                test_fcn=functools.partial(scipy.stats.ttest_rel,
                                            nan_policy="omit"),
                                p_level = 0.05,
                                compare_to="N",
                                compare_fcn=functools.partial(np.take, indices=0),
                                ylabel="", doPlots=True):
        """
        Performes the statistical test on paired comparisons

        Arguments
        results    ->   dict with keys (subj, trial), as returned
                        from a call to apply_to_all_trials. Each value in
                        the dict should be a list containing
                        [mean, std, min, Q1, Q2, Q3, max]
        title      ->   String
        test_fcn   ->   Function that computes the statistical comparisons. The
                        default is to do standard paired t-test and omitting
                        NaNs.
        compare_to ->   Test case to compare others to. Default "N"
        compare_fcn ->  Function that computes the value to use in the comparisons
                        Defaults to the first element, i.e. mean
        doPlots    ->   True means plot results
        """

        # Find the comparison data
        compData = dict( [ (subj, compare_fcn(results[(subj, trial)]))
                        for (subj, trial) in results.keys() if trial==compare_to ] )


        # The other data
        allConditions = set(["N", "B", "M", "D"])
        allOtherConditions = allConditions - set(compare_to)

        # Now run the comparison for all conditions
        allComparisons = {}
        allPvals = {}
        for cond in allOtherConditions:
            condData = dict( [ (subj, compare_fcn(results[(subj, trial)]))
                        for (subj, trial) in results.keys() if trial==cond ] )
            # Pair the data
            pairs = [(condData[subj], compData[subj])
                        for subj in condData.viewkeys() & compData.viewkeys()]

            a,b = zip(*pairs)
            t, p = test_fcn(a,b)
            allComparisons[cond + "-" + compare_to] = np.array(a) - np.array(b)
            allPvals[cond + "-" + compare_to] = p


        if doPlots:
            fig= pyplot.figure(figsize=(6,4))
            ax = fig.add_subplot(1,1,1)
            font0 = FontProperties()
            fontBold = font0.copy()
            fontBold.set_weight("bold")

            comparisons = allComparisons.keys()
            diffs = allComparisons.values()
            pvals = [allPvals[comparison] for comparison in comparisons]

            bp = ax.boxplot(diffs, showmeans=True, sym='')
            pyplot.setp(bp['boxes'], color='black')
            pyplot.setp(bp['whiskers'], color='black')
            #pyplot.setp(bp['fliers'], color='red', marker='+')
            xtickNames = pyplot.setp(ax, xticklabels=comparisons)
            pyplot.setp(xtickNames, fontsize=12)
            ax.set_title(title)
            ax.set_xlabel('Comparisons', fontsize=12)

            ax.set_ylabel(ylabel, fontsize=12)

            # Annoate with p value
            yl = ax.get_ylim()
            ppos = 1.1*yl[1]
            yl = (yl[0], 1.2*yl[1])
            ax.set_ylim(yl)
            for i in range(len(diffs)):
                if pvals < p_level:
                    txt = ax.text(i+1, ppos, "p=%1.3f" % pvals[i],
                        fontweight="bold",
                        horizontalalignment="center",
                        fontsize=13)
                else:
                    txt = ax.text(i+1, ppos, "p=%1.3f" % pvals[i],
                    horizontalalignment="center",
                    fontsize=12)


            self.savepdf(title)


    def save_statistics(self, results, title):
        """
        Saves the descriptive statistics contained in results to a file
        with the provided title and today's date.

        Arguments
        results    ->   dict with keys (subj, trial), as returned
                        from a call to apply_to_all_trials. Each value in
                        the dict should be a list containing
                        [mean, std, min, Q1, Q2, Q3, max]
        title      ->   String
        """

        from itertools import cycle

        # File to save to in folder named as today's date
        resdir = os.path.join(os.getcwd(), date.isoformat(date.today()) )
        if not os.path.exists(resdir):
            os.makedirs(resdir)


        fnameparts = title.split()
        fnameparts.append(date.today().isoformat())
        fnameparts.append("descriptive-stats")
        fnameparts.append(".xlsx")
        fname = os.path.join(resdir, "-".join(fnameparts))
        fname = fname[:-6] + fname[-5:] # Remove last hyphen before suffix

        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(fname)
        worksheet = workbook.add_worksheet()

        # Create some format objects to make the spreadsheet easier to read
        headerformat = workbook.add_format()
        headerformat.set_bold()
        headerformat.set_font_color('white')
        headerformat.set_bg_color('#990000')

        blockformat1 = workbook.add_format()
        blockformat1.set_bg_color('#AAAAAA')
        blockformat2 = workbook.add_format()
        blockformat2.set_bg_color('#CCCCCC')
        formats = cycle((blockformat1, blockformat2))

        # Unique list of subjects
        subjects = list(set([subj for (subj, trial) in results.keys()]))

        # Sort them
        subjects.sort(key=lambda id: int(id[1:]))

        # Order of conditions
        conds = ["N", "B", "M", "D"]

        #1/0

        worksheet.write(1,0, "Subject", headerformat)
        #worksheet.write(0,0, "", headerformat)
        col = 1
        for cond in conds:
            worksheet.write(0, col, cond, headerformat)
            c = 0
            for heading in ("Mean", "Stdv", "Min", "Q1", "Q2", "Q3", "Max"):
                worksheet.write(1,col+c, heading, headerformat)
                c += 1
            col += 7

        row = 2
        for subj in subjects:
            worksheet.write(row, 0, subj, headerformat)
            col = 1
            for cond in conds:
                fmt = formats.next()
                try:
                    dta = results[(subj, cond)]
                except:
                    # Assuming error occurs because data missing. Cells left blank
                    logging.warning("Exception. Missing data for" +
                        " subject %s, condition %s" % (subj, c))
                    col += 7
                    continue
                for i in range(len(dta)):
                    if not np.isnan(dta[i]):
                        worksheet.write(row, col+i, dta[i], fmt)
                col +=7
            row += 1

        workbook.close()

    def make_boxplot(self, results, title, ylim=None, ylabel="Value"):
        """ Make a boxplot, and saves the figure as a pdf file
        using the title and date.
        The results argument is a dict with keys (subj, trial), as returned
        from a call to apply_to_all_trials.

        IF results is a list, then both results are plotted in a way that
        facilitates visual comparison.
        """


        fig= pyplot.figure(figsize=(8.5,5))
        ax = fig.add_subplot(1,1,1)
        if type(results) is dict:
            self._make_boxplot_singleresults(results, title, ylim, ylabel, ax)
        else:
            # Should be list of dicts. The length of the first series will
            # determine the subjects and conditions that are plotted
            n = len(results)
            xpositions = self._make_boxplot_singleresults(results[0], title,
                                            ylim, ylabel,
                                            ax, 0.9, 0, n, annotate=False)
            for i in range(1,n):
                annotate = (i == n/2)

                self._make_boxplot_singleresults(results[i], title, ylim, ylabel,
                                            ax, 0.9 - 0.4*float(i+1)/float(n),
                                            i, n, xpositions, annotate)
        fname = title
        self.savepdf(fname)

    def savepdf(self, fname=""):
            pyplot.draw()

            # Save figure
            resdir = os.path.join(os.getcwd(), date.isoformat(date.today()) )
            if not os.path.exists(resdir):
                os.makedirs(resdir)


            fnameparts = fname.split()
            fnameparts.append(date.today().isoformat())
            fnameparts.append(".pdf")
            figname = os.path.join(resdir, "-".join(fnameparts))
            figname = figname[:-5] + figname[-4:] # Remove last hyphen before suffix
            pyplot.savefig(figname, format="pdf")


    def _make_boxplot_singleresults(self, results, title,  ylim, ylabel, ax,
                                        graylevel=0.9, serie=0, nseries=1,
                                        xpositions=None, annotate=True):
        """
        Generates a boxplot given the axes to plot

        Arguments
        results       ->  dict with data indexed by the tuple (subj, trial)
        title         ->  String
        ylim          ->  tuple. Can be None
        ylabel        ->  String
        ax            ->  An Axes object to draw plot in
        graylevel     ->  Fill color of boxes
        serie         ->  The index of this series if set of results
        nseries       ->  Number of series in the set
        xpositions    ->  dict indexed by the tuple (subj, trial) giving the
                          x positions of the boxes
        annotate      ->  If True will write xticklabels and subject name
        """
        # Unique list of subjects
        subjects = list(set([subj for (subj, trial) in results.keys()]))

        # Sort them
        subjects.sort(key=lambda id: int(id[1:]))

        # Order of conditions
        conds = ["N", "B", "M", "D"]


        ns = len(subjects)
        k=-1
        subjdta =  []
        pos = []
        subjannot = []
        midp = []
        maxp = -1e10
        xticknames = []
        if serie == 0:
            xpositions = {}
        for subj in subjects:
            k += 1
            i = 0
            subjpos = []
            for c in conds:
                i += 1
                try:
                    dta = results[(subj, c)]
                except:
                    # Assuming error occurs because data missing. Put in a few nans
                    logging.warning("Exception. Missing data")
                    continue
                if len(dta) == 0:
                    continue

                if serie == 0:
                    xpos = (5*nseries)*k+(i*nseries)
                    xpositions[(subj, c)] = xpos
                else:
                    try:
                        xpos = xpositions[(subj,c)] + serie
                    except KeyError:
                        #print "Position for subj %s, condition %s not found" % (subj, c)
                        # subject, condition not found. Skip
                        continue

                # Remove any nans
                dta = [el_ for el_ in dta if not np.isnan(el_)]
                if len(dta) == 0:
                    continue
                subjdta.append(dta)
                pos.append(xpos)
                subjpos.append(xpos)
                maxp = max(maxp, np.max(dta))
                xticknames.append(c)
            if len(subjpos) > 0:
                midp.append(np.mean(np.array(subjpos))-1)
                subjannot.append(subj)

        bp = ax.boxplot(subjdta, positions=pos, sym='', patch_artist=True)
        pyplot.setp(bp['boxes'], color='black')
        for patch in bp['boxes']:
            patch.set_facecolor("%f" % graylevel)

        pyplot.setp(bp['whiskers'], color='black')
        #pyplot.setp(bp['fliers'], color='red', marker='+')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Subjects and conditions', fontsize=12)

        ax.set_ylabel(ylabel, fontsize=12)

        yl = ax.get_ylim()
        yl = (yl[0], maxp + 0.1*(yl[1]-yl[0]))

        if ylim is None:
            ax.set_ylim(yl)
        if ylim is not None:
            ax.set_ylim(ylim)
            maxp = ylim[1] - 0.2*(ylim[1]-ylim[0])

        if annotate:
            if not nseries % 2:
                # Even number of series, so ticks between boxes
                locs, labels = pyplot.xticks()
                pyplot.setp(ax, xticks=locs-0.5)

            locs, labels = pyplot.xticks()
            xtickrange = max(locs) - min(locs)
            ax.set_xlim((min(locs) - 0.08*xtickrange, max(locs) + 0.08*xtickrange))

            xtickNames = pyplot.setp(ax, xticklabels=xticknames)
            pyplot.setp(xtickNames, fontsize=10)
            # Annoate with subject id
            for (subj_, pos_) in itertools.izip(subjannot, midp):
                ax.text(pos_, maxp, subj_)

        return xpositions

    def fix_cycle_events(self, subject, trial, k=2):
        """ Checks the PNAtICLA attribute, computes the
        0.25, 0.5 and 0.75 quantiles. Determines then the start and end of each cycle
        so that only cycles with length that is less than median + k*interquartiledistance
        are kept.
        The start and end events are recorded in the attribute 'PNAtCycleEvents' as
        a list of two-tuples.
        """

        subj = self.hdfFile[subject]
        tr = subj[trial]
        ics =tr.attrs["PNAtICLA"]
        cycles = kinematics.fix_cycles(ics, k)
        tr.attrs["PNAtCycleEvents"] = cycles
        self.hdfFile.flush()

        logging.info("Subject %s, trial %s: %d of %d cycles discarded (%2.1f %%)" \
            % (subject, trial, (len(ics)-len(cycles)), len(ics),
               float(len(ics)-len(cycles))/len(ics)*100) )


    def has_trial_attribute(self, subject, trial, attrname):
        f = self.hdfFile
        subj = f[subject]
        tr = subj[trial]
        if attrname in tr.attrs.keys():
            print "Trial %s of subject %s contains attribute %s" % (trial, subject, attrname)
            return True
        else:
            print "Trial %s of subject %s DOES NOT contain attribute %s" % (trial, subject, attrname)
            return False

    def plot_imu_data(self, subject="S7", trial="D", imu="N", SIUnits=False):
        f = self.hdfFile
        g = f[subject]
        sg = g[trial]
        ds = sg[imu]

        if SIUnits:
            gyrodta = np.array(ds[:,1:4])*np.pi/180.0
            gyrolabel = "Gyro [rad/s]"
            accdta = np.array(ds[:,4:7])*9.82
            acclabel = "Acc [m/s^2]"
        else:
            gyrodta = ds[:,1:4]
            gyrolabel = "Gyro [deg/s]"
            accdta = ds[:,4:7]
            acclabel = "Acc [g]"

        pyplot.figure(figsize=(12,10))

        pyplot.subplot(3,1,1)
        pyplot.plot(ds[:,0], gyrodta)
        pyplot.ylabel(gyrolabel)

        pyplot.subplot(3,1,2)
        pyplot.plot(ds[:,0], accdta)
        pyplot.ylabel(acclabel)

        pyplot.subplot(3,1,3)
        pyplot.plot(ds[:,0], ds[:,7:10])
        pyplot.ylabel("Magn [G]")
        pyplot.legend(("x", "y", "z"))

        pyplot.title("IMU data for subject %s, trial %s, imu %s" % (subject, trial, imu))
        pyplot.show()

        return ds

    def detect_steps(self, subject="S7", trial="D"):
        """ Loads the data from the LA and RA imus, and look for peaks in the data
        above a given treshold.
        """
        subj = self.hdfFile[subject]
        tr = subj[trial]
        mindist = 240 # Hardcoded min step length
        for ankle in ("LA", "RA"):
            ankledata = tr[ankle]
            accmagn = np.sqrt(np.sum(ankledata[:,4:7]**2, axis=1))

            figh = pyplot.figure()
            pyplot.plot(ankledata[:,0], accmagn)
            pyplot.title("Vertical acc for %s in trial %s of subject %s" % (ankle,trial,subject))
            answer = raw_input("Threshold to use?: ")
            thr = float(answer)

            abovethreshold = np.nonzero(accmagn > thr)
            # Find peaks in the continues blocks of data above threshold
            peaks = []
            currentpeak = 0
            currentpeakind = 0
            prevind = 0
            prevpeakind = 0
            for ind in abovethreshold[0]:
                if ind > prevind+1:
                    # Check that it is not a double peak in the step
                    if currentpeakind > prevpeakind + mindist:
                        peaks.append(currentpeakind)
                        prevpeakind = currentpeakind

                    currentpeak = 0
                if accmagn[ind] > currentpeak:
                  currentpeak = accmagn[ind]
                  currentpeakind = ankledata[ind,0]
                prevind = ind

            yl = [0, 6]
            for ind in peaks:
                pyplot.plot([ind, ind], yl, 'm')

            pyplot.title("Initial contact in data for " + ankle)

            tr.attrs["PNAtIC"+ankle] = peaks
        self.hdfFile.flush()

    def get_cycle_frequency(self, subject="S7", trial="D",\
                                startTime=5*60,\
                                anTime=60, doPlots=True):
        """ Will determine the cycle frequency for the given trial, assuming that
        the start of each cycle is detected and available in the trial attribute 'PNAtICLA'.
        The result will be stored in the trial attribute 'cycleFrequency' as an array,
        and returned to the caller.

        """
        [imudta, tr, subj] = self.get_imu_data(subject, trial, "LA", startTime, anTime)
        firstPN = imudta[0,0]
        lastPN = imudta[-1,0]

        try:
            cycledta = self.get_cycle_data(subject, trial, "LA", firstPN, lastPN)
        except KeyError:
            logging.warning("No step cycles found yet. Run detect_steps first!")
            return

        #dt = 1.0/256
        dt = 1.0/262.0 # Weird, but this is the actual sampling period

        #periods = [(cPN-pPN)*dt for (cPN, pPN) in itertools.izip(cycledta[1:], cycledta[:-1])]
        periods = [(stop_-start_)*dt for (start_, stop_) in cycledta]
        freqs = [1.0/p for p in periods]

        subj = self.hdfFile[subject]
        tr = subj[trial]
        #tr.attrs['cycleFrequency'] = freqs

        if doPlots:
            # Make a histogram of the frequencies
            #binsize = 0.1
            #bins = defaultdict(int)
            #for fr in freqs:
            #   bin = int(fr/binsize)
            #    bins[bin] += 1
            pyplot.figure()
            pyplot.hist(freqs, bins=10)
            pyplot.xlabel("Frequency [Hz]")
            pyplot.title("Cycle frequencies for subj %s, trial %s"\
                             % (subject, trial))

        return freqs

    def get_rotation(self, subject="S7", trial="D", imu="LH",
                            startTime=3*60, anTime=60, phiOnly=True):
        """
        Calculates the rotation of the imu wrt the orientation at the beginning
        of each step.
        Returns
        phi       <-  angle of rotation
        q         <-  Quaternion array containing the complete rotation
        if phiOnly is true, returns only phi (suprise).
        """
        [imudta, s_, tr_] = self.get_imu_data(subject, trial, imu,
                                            startTime, anTime, split=True,
                                            SIUnits=True)

        phi = []
        q = []
        for imudta_ in imudta:
            gyro = imudta_[:,1:4]
            acc = imudta_[:,4:7]
            mag = imudta_[:,7:10]
            tvec = imudta_[:,-1]
            qe, theta = self.rotationEstimator(tvec, gyro, acc, mag)
            q.append(qe)
            #phi.append([np.arccos(q_.w) for q_ in qe])
            phi.append(np.arccos(qe.w))
        if phiOnly:
            return phi
        else:
            return (phi, q)

    def get_ROM_joint_angle(self, subject="S7", trial="D", imus=["LA", "LT"],
                           startTime=5*60, anTime=60, doPlots=True):
        mma = self.get_minmax_joint_angle(subject, trial, imus,
                                          startTime, anTime, False)

        rom = [(maxs-mins)*180/np.pi for (mins, maxs) in mma]
        if doPlots:
            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(rom, bins=10)
            pyplot.xlabel("Angle")
            pyplot.title("Range of motion in joint (%s, %s) for subj %s, trial %s"\
                             % (imus[0], imus[1], subject, trial))


        return rom

    def get_minmax_joint_angle(self, subject="S7", trial="D", imus=["LA", "LT"],
                           startTime=5*60, anTime=60, doPlots=True):

        # First see if angle exists? Not at the moment. Implement later if needed

        ka = self.get_angle_between_segments(subject, trial, imus,
                                        startTime, anTime, False)

        minmaxangle =  [(np.amin(angle), np.amax(angle)) for angle in ka]

        if doPlots:
            fig = pyplot.figure()
            ax = fig.add_subplot(1,2,1)
            ax.hist([mins*180/np.pi for (mins, maxs) in minmaxangle], bins=10)
            pyplot.xlabel("Angle")
            pyplot.title("Minimum joint angle (%s, %s) for subj %s, trial %s"\
                             % (imus[0], imus[1], subject, trial))
            ax = fig.add_subplot(1,2,2)
            ax.hist([maxs*180/np.pi for (mins, maxs) in minmaxangle], bins=10)
            pyplot.xlabel("Angle")
            pyplot.title("Maximum joint angle (%s, %s) for subj %s, trial %s"\
                             % (imus[0], imus[1], subject, trial))


            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist([(maxs-mins)*180/np.pi for (mins, maxs) in minmaxangle], bins=10)
            pyplot.xlabel("Angle")
            pyplot.title("Range of motion in knee joint for subj %s, trial %s"\
                             % (subject, trial))


        return minmaxangle

    def get_angle_between_segments(self, subject="S7", trial="D", imus=["LA", "LT"],
                                   startTime=5*60, anTime=60,
                                   jointAxes=None, useEKF=True, doPlots=False):
        """ Calculates the angle between the two IMUs. If the argument jointAxis
        is provided, this may be a (2x3) numpy array with the joint
        axis direction in the local frames of the two IMUs, respectively, or if
        empty, the directions are estimated from the data. The
        angle is then calculated by projecting the angular velocity onto the
        plane normal to the joint axes, and integrating the difference.
        The integration is restarted at each cycle start.
        If jointAxis is None, then the angle to the vertical is calculated,
        and then the difference in the angle to vertical is returned.
        If useEKF is True, then an ekf is used to track the angle to the vertical.
        """

        if jointAxes is not None:
            if jointAxes == []:
                # Estimate the axes
                (d0, d1) = self.estimate_joint_axes(subject, trial, imus,
                                                startTime, anTime)
            else:
                d0 = jointAxes[0]
                d1 = jointAxes[1]

            logging.DEBUG("Joint axes used in the calculation")
            logging.DEBUG(d0)
            logging.DEBUG(d1)

            if useEKF:
                angleTracker0 = angle_to_vertical_ekf_tracker(sagittalDir=d0,
                                                        var_angvel=1e-2,
                                                        var_incl=1e-1,
                                                        m=20)
                angleTracker1 = angle_to_vertical_ekf_tracker(sagittalDir=d1,
                                                        var_angvel=1e-2,
                                                        var_incl=1e-1,
                                                        m=20)
                a2v0 = self.get_angle_to_vertical(subject, trial, imus[0],
                                              startTime, anTime,
                                              doPlots=False,
                                              angleTracker=angleTracker0)
                a2v1 = self.get_angle_to_vertical(subject, trial, imus[1],
                                              startTime, anTime,
                                              doPlots=False,
                                              angleTracker=angleTracker1)

                angleBetweenSegments = []

                x = np.linspace(0,99, 100)
                for (a0, a1) in itertools.izip(a2v0, a2v1):
                    # Normalize to 100 data points, then compute the difference
                    a0f = a0.flatten()
                    x0 = np.linspace(0,99, len(a0f))
                    f0 = interp1d(x0, a0f, kind='linear')
                    a0i = f0(x)

                    a1f = a1.flatten()
                    x1 = np.linspace(0,99, len(a1f))
                    f1 = interp1d(x1, a1f, kind='linear')
                    a1i = f1(x)

                    angleBetweenSegments.append(a0i-a1i)

            else:

                [imudta0, s_, tr_] = self.get_imu_data(subject, trial, imus[0],
                                                startTime, anTime, split=True)
                [imudta1, s_, tr_] = self.get_imu_data(subject, trial, imus[1],
                                                startTime, anTime, split=True)

                sfreq = self.hdfFile.attrs['packetNumbersPerSecond']
                dt = 1.0/sfreq

                angleBetweenSegments = []
                x = np.linspace(0,99, 100)
                for (dta0_, dta1_) in itertools.izip(imudta0, imudta1):
                    g0 = dta0_[:,1:4]*np.pi/180.0
                    g1 = dta1_[:,1:4]*np.pi/180.0

                    t0_ = (dta0_[:,0] - dta0_[0,0])*dt
                    t1_ = (dta1_[:,0] - dta1_[0,0])*dt
                    if t0_[-1] > t1_[-1]:
                        # Resample the longer data set to the shorter
                        f_ = interp1d(t0_,g0, kind='linear', axis=0)
                        g0_ = f_(t1_)
                        g1_ = g1
                        tvec = t1_
                    else:
                        f_ = interp1d(t1_, g1, kind='linear', axis=0)
                        g1_ = f_(t0_)
                        g0_ = g0
                        tvec = t0_

                    ja = cpkinematics.planar_joint_angle_integrated(g0_, g1_,
                                                                    tvec,
                                                                    d0, d1)

                    # Normalize to 100 data points,
                    x0 = np.linspace(0,99, len(ja))
                    f0 = interp1d(x0, ja, kind='linear')
                    angleBetweenSegments.append(f0(x))

        else:
            a2v0 = self.get_angle_to_vertical(subject, trial, imus[0],
                                          startTime, anTime, False)
            a2v1 = self.get_angle_to_vertical(subject, trial, imus[1],
                                          startTime, anTime, False)

            angleBetweenSegments = []

            x = np.linspace(0,99, 100)
            for (a0, a1) in itertools.izip(a2v0, a2v1):
                # Normalize to 100 data points, then compute the difference
                a0f = a0.flatten()
                x0 = np.linspace(0,99, len(a0f))
                f0 = interp1d(x0, a0f, kind='linear')
                a0i = f0(x)

                a1f = a1.flatten()
                x1 = np.linspace(0,99, len(a1f))
                f1 = interp1d(x1, a1f, kind='linear')
                a1i = f1(x)

                angleBetweenSegments.append(a0i-a1i)


        if doPlots and self.debug:
            pyplot.figure()
            for a in angleBetweenSegments:
                pyplot.plot(a*180/np.pi)
            pyplot.title("Angle between imus %s and %s for subj %s, trial %s"\
                             % (imus[0], imus[1], subject, trial))
            pyplot.ylabel('Degrees')

        #subj = self.hdfFile[subject]
        #tr = subj[trial]
        #try:
        #    abi = tr.attrs['angleBetweenSegments']
        #except KeyError:
        #    abi = []

        #abi.append( ( (imus[0], imus[1]), angleBetweenSegments) )

        return angleBetweenSegments

    def get_angle_between_segments_markers(self, subject="S4", trial="D", imus=["LA", "LT"],
                                   startTime=5*60, anTime=60, doPlots=False):
        """
        Same as get_angle_between_segments but using marker data. This is only possible
        for trials "D" and "N", and for imus LA, LT, N, LH. If called for other
        trial or imu, returns empty list.

        Sagittal plane motion is assumed. The angle to the vertical is calculated
        for both segments, and then the angle is defined as the angle of the
        first segment in the list minus the angle of the second.

        Returns a list of scalar time series, one per gait cycle.
        """

        a2v0 = self.get_angle_to_vertical_markers(subject, trial, imus[0],
                                          startTime, anTime)
        a2v1 = self.get_angle_to_vertical_markers(subject, trial, imus[1],
                                          startTime, anTime)

        angleBetweenSegments = []

        x = np.linspace(0,99, 100)
        for (a0, a1) in itertools.izip(a2v0, a2v1):
            # Normalize to 100 data points, then compute the difference
            a0f = a0.flatten()
            x0 = np.linspace(0,99, len(a0f))
            f0 = interp1d(x0, a0f, kind='linear')
            a0i = f0(x)

            a1f = a1.flatten()
            x1 = np.linspace(0,99, len(a1f))
            f1 = interp1d(x1, a1f, kind='linear')
            a1i = f1(x)

            angleBetweenSegments.append(a0i-a1i)


        if doPlots and self.debug:
            pyplot.figure()
            for a in angleBetweenSegments:
                pyplot.plot(a*180/np.pi)
            pyplot.title("Angle between imus %s and %s for subj %s and trial %s, calculated from marker data"\
                             % (imus[0], imus[1], subject, trial))
            pyplot.ylabel('Degrees')


        return angleBetweenSegments

    def get_minmax_angle_to_vertical(self, subject="S7", trial="D", imu="N",
                           startTime=5*60, anTime=60, doPlots=True):

        va = self.get_angle_to_vertical(subject, trial, imu, startTime,
                                        anTime, False)
        # va is a list of np arrays
        minmaxangle =  [(np.amin(angle), np.amax(angle)) for angle in va]

        return minmaxangle

    def get_RoM_angle_to_vertical(self, subject="S7", trial="D", imu="N",
                           startTime=5*60, anTime=60, doPlots=True):

        mma = self.get_minmax_angle_to_vertical(subject, trial, imu,
                                                startTime, anTime, False)

        rom = [(maxs-mins)*180/np.pi for (mins, maxs) in mma]
        if doPlots:
            fig = pyplot.figure()
            ax = fig.add_subplot(1,1,1)
            ax.hist(rom, bins=10)
            pyplot.xlabel("Angle")
            pyplot.title("Range of angle to vertical in imu %s for subj %s, trial %s"\
                             % (imu, subject, trial))


        return rom

    def get_angle_to_vertical(self, subject="S7", trial="D", imu="LT",
                        startTime=5*60, anTime=4*60, sagittalPlane=False,
                        doPlots=True,
                        angleTracker=None):
        """
         Tracks the orientation, finds the direction of the vertical, and
        calculates the angle to the vertical. If sagittalPlane is not None,
        then then the angle is calculated in the sagittal plane. There are two
        ways to achieve this: 1) sagittalPlane is a unit 3d vector giving the
        direction normal to the sagittal plane; 2) sagittalPlane is the name
        of another imu attached to an adjoining segment. In the latter case,
        the joint axes are calculated.
        If resetAtIC is True, then the tracking is restarted at each initial
        contact.
        """

        if angleTracker is None:
            angleTracker = self.rotationEstimator

        [imudta, s_, tr_] = self.get_imu_data(subject, trial, imu,
                                            startTime, anTime, split=True,
                                            SIUnits=True)
        (gyroref, accref, magref) = self.get_reference_imu_data(subject, imu)

        a2v = []

        for imudta_ in imudta:
            gyro = imudta_[:,1:4]
            acc = imudta_[:,4:7]
            mag = imudta_[:,7:10]
            tvec = imudta_[:,-1]
            qe, theta = angleTracker(tvec, gyro, acc, mag, gyroref, accref, magref)
            a2v.append(theta)
            #
            # [qimu, imudta] = self.track_orientation(subject,
            #                                         trial, imu,
            #                                         startTime, anTime, False)
            #
            # [imuDisp, imuVel, imuGvec] = self.track_displacement(subject,
            #                                                       trial,
            #                                                       imu,
            #                                                       startTime,
            #                                                       anTime,
            #                                                       qimu=qimu)
            #
            # a2v = []
            # for (q_, g_) in itertools.izip(qimu, imuGvec):
            #     # g_ is in static frame coinciding with imu-frame at start of
            #     # each cycle.
            #     # The longitudinal direction of the segment is defined by the
            #     # x-axis (pointing downward)
            #     qa = quat.QuaternionArray(q_[:,1:5])
            #     g_.shape = (3,1)
            #     x_ = np.array([[-1., 0, 0]]).T
            #     imux_ = qa.rotateFrame(x_)
            #
            #     if sagittalPlane:
            #         imux_[2,:] = 0.0
            #         g_[2,] = 0.0
            #
            #     # Find the signed angle between two vectors
            #     gnrm = np.sqrt(np.sum(g_**2))
            #     gnormed = g_ / gnrm
            #     xdotg = np.dot(imux_.T, gnormed)
            #     xdotg = xdotg.flatten() / np.sqrt(np.sum(imux_**2, 0))
            #
            #     # Make sure acos will work
            #     xdotg[np.nonzero(xdotg>1)] = 1.
            #     xdotg[np.nonzero(xdotg<-1)] = -1.
            #
            #     gcross = np.cross(imux_.T, gnormed.T)
            #     sgn = np.sign(gcross[:,2])
            #
            #     a2v.append(np.arccos(xdotg) * sgn.T)
            #

        if self.debug or doPlots:
            pyplot.figure()
            for a in a2v:
                pyplot.plot(a*180/np.pi)
            pyplot.title("Angle to vertical for subj %s, trial %s, imu %s"\
                             % (subject, trial, imu))

        #subj = self.hdfFile[subject]
        #tr = subj[trial]
        #try:
        #        angle2vertical = tr.attrs['angle2vertical']
        #        except KeyError:
        #    angle2vertical = []

        #angle2vertical.append((imu, a2v))

        return a2v


    def get_angle_to_vertical_markers(self, subject="S6", trial="D", imu="LH",
                        startTime=5*60, anTime=4*60, vertDir=[0.,0., 1],
                        sagittalDir=[-1.0, 0, 0]):
        """
        Same as get_angle_to_vertical but using marker data. This is only possible
        for trials "D" and "N", and for imus LA, LT, N, LH. If called for other
        trial or imu, returns empty list.

        The calculation is based on the markers referred to as "upper" and "lower"
        for the IMU. The vector from lower to upper is calculated, and from its
        displacements, the vector normal to the plane of motion is found. It
        is assumed that this direction is approximately in the direction provided
        by the argument sagittalDir. The angle to the vertical is positive for
        a rotation about a vector pointing to the left.

        Returns a list of scalar time series, one per gait cycle.
        """

        upperName = IMU_MARKERS[imu]["upper"]
        lowerName = IMU_MARKERS[imu]["lower"]

        #try:
        mdta = self.get_marker_data(subject, trial,
                    markers=[upperName, lowerName], startTime=startTime,
                    anTime=anTime)
        #except ValueError:
        #    return []

        upper = mdta[upperName]
        lower = mdta[lowerName]

        # change zeros into NaNs
        missingupper,  = np.where(np.sum(upper, axis=1) == 0)
        upper[missingupper, :] = np.nan
        missinglower, = np.where(np.sum(lower, axis=1) == 0)
        lower[missinglower, :] = np.nan

        angle = kinematics.angle_to_vertical(upper, lower, vertDir=vertDir,
                                            sagittalDir=sagittalDir)

        # Split data into cycles
        a2v = [angle[startInd_:stopInd_]
                            for (startInd_, stopInd_) in mdta["cycledataInd"]]

        if self.debug:
            pyplot.figure()
            for a in a2v:
                pyplot.plot(a*180/np.pi)
            pyplot.title("Angle to vertical for subj %s, trial %s, markers %s and %s"\
                             % (subject, trial, upperName, lowerName))

        return a2v



    def get_sagittal_plane_displacement(self, subject, trial, imu,
                                        startTime, anTime, doPlots=True,
                                        displacementTracker=None,
                                        resetAtIC=True,
                                        g=9.82, gThreshold=5e-1):
        """
        Tracks the displacement in the sagittal plane. The definition of the
        sagittal plane and vertical reference direction is contained in the
        displacementTracker (see class sagittal_plane_displacement_tracker).
        If resetAtIC is True, then the tracking is restarted at each initial
        contact.
        """

        if resetAtIC:
            [imudta, s_, tr_] = self.get_imu_data(subject, trial, imu,
                                            startTime, anTime, split=True,
                                            SIUnits=True)
            displacement = []
            acceleration = []
            accdata = []
            for imudta_ in imudta:
                gyro = imudta_[:,1:4]
                acc = imudta_[:,4:7]
                tvec = imudta_[:,-1]

                d_, a_, a0_ = displacementTracker(tvec, acc, gyro,
                                                g=g, gThreshold=gThreshold,
                                                plotResults=doPlots)
                displacement.append(d_)
                acceleration.append(a_)
                accdata.append(a0_)

        else:
            [imudta, s_, tr_] = self.get_imu_data(subject, trial, imu,
                                        startTime, anTime, split=False,
                                        SIUnits=True)
            gyro = imudta[:,1:4]
            acc = imudta[:,4:7]
            tvec = imudta[:,-1]
            d_, a_, a0_  = displacementTracker(tvec, acc, gyro,
                                        g=g, gThreshold=gThreshold,
                                        plotResults=doPlots)
            # Split into cycles
            [imudtaLA, s_, tr_] = self.get_imu_data(subject, trial, "LA",
                                        startTime, anTime, split=False)
            firstPN = imudtaLA[0,0]
            lastPN = imudtaLA[-1,0]
            cyclePNs = self.get_cycle_data(subject, trial, imu, firstPN, lastPN)
            packetNumbers = np.asarray(imudta[:,0], np.int32)

            cycledtaInds = [ (np.where(packetNumbers <= cd_[0])[0][-1],
                              np.where(packetNumbers >= cd_[1])[0][0])
                                for cd_ in cyclePNs ]
            displacement = [ d_[startInd_:stopInd_,:]
                                for (startInd_, stopInd_) in cycledtaInds ]
            acceleration = [ a_[startInd_:stopInd_, :]
                                for (startInd_, stopInd_) in cycledtaInds ]
            accdata = [ a0_[startInd_:stopInd_,:]
                    for (startInd_, stopInd_) in cycledtaInds ]

        if doPlots:
            pyplot.figure()
            for d_ in displacement:
                pyplot.plot(d_[:,0], d_[:,1])

            pyplot.title("Displacement in the sagittal plane subj %s, trial %s, imu %s"
                             % (subject, trial, imu))
            pyplot.figure()
            for (a_, acc_) in itertools.izip(acceleration, accdata):
                pyplot.subplot(211)
                pyplot.plot(a_[:,0], linewidth=2)
                pyplot.plot(acc_[:,0])
                pyplot.subplot(212)
                pyplot.plot(a_[:,1], linewidth=2)
                pyplot.plot(acc_[:,1])


            pyplot.title("acceleration in the sagittal plane subj %s, trial %s, imu %s"
                             % (subject, trial, imu))

        return displacement


    def get_range_of_motion(self, subject="S7", trial="D", imu="LA",
                            startTime=5*60,
                            anTime=60, doPlots=True):
        """ Will track the orientation of the imu, and then
        calculate the range of motion in degrees.
        """
        [qimu, cycledta, imudta, cycleinds] = self.track_orientation(subject, trial, imu,
                                                          startTime, anTime, False)

        def _rotation(q):
            q0 = quat.Quaternion(q[0,1], q[0,2], q[0,3], q[0,4]).conjugate
            rot = np.zeros(q.shape[0])
            rad2deg = 180.0/np.pi
            for i in range(q.shape[0]):
                qi = quat.Quaternion(q[i,1], q[i,2], q[i,3], q[i,4]) * q0
                rot[i] = 2*math.acos(qi.w)*rad2deg
            return rot

        RoM = []
        rots = []
        for q_ in qimu:
            rot = _rotation(q_)
            rots.append(rot)
            RoM.append(np.max(rot))

        if doPlots:
            pyplot.figure()
            pyplot.hist(RoM, bins=20)
            pyplot.title("Range of motion in degrees for subj %s, trial %s, imu %s"\
                             % (subject, trial, imu))

        subj = self.hdfFile[subject]
        tr = subj[trial]
        imud = tr[imu]
        imud.attrs['rangeOfMotion'] = RoM


        return RoM

    def get_vertical_displacement(self, subject="S7", trial="D", imu="LA",\
                                      startTime=5*60,\
                                      anTime=60, doPlots=True):
        """
        First tracks the displacement using self.track_displacement. Then
        calculates the vertical displacement by projecting the displacement
        onto the vertical direction in all cycles.

        Returns
        vDisps    <-  List with (N,) arrays containing the vertical displacement
                      for each gait cycle
        """

        [imuDisp, imuVel, imuGvec, imuSagDir] = self.track_displacement(
                                subject, trial, imu, startTime, anTime, doPlots)

        vDisps = [ np.dot(d_, g_) for (d_, g_) in
                                            itertools.izip(imuDisp, imuGvec)]

        if doPlots and self.debug:
            pyplot.figure()
            for vd_ in vDisps:
                pyplot.plot(vd_)

            pyplot.title("Vertical displacment for subj %s, trial %s, imu %s"\
                             % (subject, trial, imu))

        return vDisps

    def get_vertical_displacement_markers(self, subject="S4", trial="D", imu="LA",\
                                      startTime=5*60,\
                                      anTime=60, doPlots=True):
        """
        Same as get_vertrack_displacement but using marker data. This is only possible
        for trials "D" and "N", and for imus LA, LT, N, LH. If called for other
        trial or imu, returns empty list.

        The calculation is based on the marker referred to as "closest"
        for the IMU. It is assumed that this marker is close to the IMU. The
        z-component of this marker is returned.

        Returns
        vDisps    <-  List with (N,) arrays containing the vertical displacement
                      for each gait cycle
        """

        disp =  self.track_displacement_markers( subject, trial, imu,
                            startTime=startTime, anTime=anTime, doPlots=False)

        vDisps = [ d_[:,2]-d_[0,2] for d_ in disp ]

        if doPlots and self.debug:
            pyplot.figure()
            for vd_ in vDisps:
                pyplot.plot(vd_)

            pyplot.title("Vertical displacment from marker data for subj %s, trial %s, imu %s"\
                         % (subject, trial, imu))

        return vDisps



    def track_displacement(self, subject="S7", trial="D", imu="LA",\
                              startTime=5*60,\
                              anTime=60, doPlots=True, qimu=None):
        """ Will first track the orientation and the displacement of the imu,
        restarting the tracking at the beginning of each step. The direction of
        gravity is identified and the displacement is corrected for the apparent
        gravitational acceleration. The resulting displacement is returned.
        Note that a body-fixed (imu-fixed) coordinate system is used.

        Returns:
        displ    <- List of (N,3) numpy arrays with displacements in the frame of
                    the IMU at start of cycle
        vel      <- As displ, but velocities
        gvecs    <- List of (3,) numpt arrays giving the vertical direction
        sagvecs  <- List of (3,) numpt arrays giving the sagittal direction. This
                    points left for all segments.
        """

        if qimu is None:
            [qimu, imudta] = self.track_orientation(subject, trial, imu,
                                                      startTime, anTime, False)
        else:
            [imudta, s_, tr_] = self.get_imu_data(subject, trial, imu,
                                                    startTime, anTime,
                                                    split=True, SIUnits=True)
        (gyroref, accref, magref) = self.get_reference_imu_data(subject, imu)

        dimu = []
        vimu = []
        gvecs = []
        sagvecs = [] # Sagittal direction unit vectors
        for (imudta_, qimu_) in itertools.izip(imudta, qimu):
            accimu = imudta_[:, 4:7]
            gyroimu = imudta_[:, 1:4]
            magimu = imudta_[:,7:10]
            tvec = imudta_[:, -1]
            [dimu_, vimu_, g_, sd_] = self.displacementEstimator(tvec, accimu,
                                            gyroimu, magimu, qimu_,
                                            accref, gyroref, magref)

            dimu.append(dimu_)
            vimu.append(vimu_)
            gvecs.append(g_)
            sagvecs.append(sd_)


        if doPlots and self.debug: # Check results
            pyplot.figure()
            pyplot.subplot(2,1,1)
            for d_ in dimu:
                pyplot.plot(d_[:,0], 'b')
                pyplot.plot(d_[:,1], 'g')
                pyplot.plot(d_[:,2], 'r')

            pyplot.title("Displacement for imu %s,  subj %s, trial %s"\
                 % (imu, subject, trial))
            pyplot.legend(("x", "y", "z"))
            yl = (-0.3, 0.3)

            pyplot.subplot(2,1,2)
            for v_ in vimu:
                pyplot.plot(v_[:,0], 'b')
                pyplot.plot(v_[:,1], 'g')
                pyplot.plot(v_[:,2], 'r')
            pyplot.title('velocity')
            yl = (-2, 2)

        return (dimu, vimu, gvecs, sagvecs)


    def track_displacement_markers(self, subject="S4", trial="D", imu="B",
                                startTime=5*60,anTime=60, doPlots=False):
        """
        Same as track_displacement but using marker data. This is only possible
        for trials "D" and "N", and for imus LA, LT, N, LH. If called for other
        trial or imu, returns empty list.

        The calculation is based on the marker referred to as "closest"
        for the IMU. It is assumed that this marker is close to the IMU. The
        position of this marker is returned.

        Returns a list of vector time series, one per gait cycle.
        """

        centerName = IMU_MARKERS[imu]["closest"]

        #try:
        mdta = self.get_marker_data(subject, trial,
                    markers=[centerName], startTime=startTime,
                    anTime=anTime)
        center = mdta[centerName]

        # change zeros into NaNs
        missing,  = np.where(np.sum(center, axis=1) == 0)
        center[missing, :] = np.nan

        # Split data into cycles
        d = [center[startInd_:stopInd_]
                        for (startInd_, stopInd_) in mdta["cycledataInd"]]

        if self.debug and doPlots:
            pyplot.figure()
            for d_ in d:
                pyplot.plot(d_[:,0], 'b')
                pyplot.plot(d_[:,1], 'g')
                pyplot.plot(d_[:,2], 'r')

            pyplot.title("Displacement for marker %s,  subj %s, trial %s"\
                         % (centerName, subject, trial))
            pyplot.legend(("x", "y", "z"))

        return d



    def track_orientation(self, subject="S7", trial="D", imu="LA",\
                              startTime=5*60,\
                              anTime=60, doPlots=True):
        """ Will track the orientation of the imu. Assumes that the start of each cycle
        is detected and available in the trial attribute 'PNAtICLA'.
        The tracking algorithm is restarted at each step. So, for each timestep,
        a quaternion is estimated that describes the orientation of the IMU w.r.t. the
        orientation at the initical contact of the current gait cycle.

        Returns:
        imuq        <- List of quaternions for each gait cycle
        imudtaSplit <- IMU data split in cycles
        """

        [imudta, tr, subj] = self.get_imu_data(subject, trial, imu,
                                startTime, anTime,
                                split=True, SIUnits=True)

        imuq = []
        for dta_ in imudta:
            gyro = dta_[:,1:4]
            acc = dta_[:,4:7]
            mag = dta_[:,7:10]
            tvec = dta_[:,-1]

            qE = self.rotationEstimator(tvec, gyro, acc, mag)

            imuq.append(qE[0])

        return [imuq, imudta]


    def get_cycle_data(self, subject, trial, imu, firstPN, lastPN):
        """
        Returns a list with tuples (startcyclePN, stopcyclePN) for all cycles
        that are within the time interval (firstPN, lastPN). The packet numbers
        firstPN and lastPN refer to packet numbers for the LA imu, which is the
        one used in detecting steps.

        """
        syncimu = self.get_PN_at_sync(subject,imu)
        syncLA = self.get_PN_at_sync(subject,"LA")
        subj = self.hdfFile[subject]
        tr = subj[trial]

        #cycledtaold = [ind-syncLA[0]+syncimu[0] for ind in tr.attrs["PNAtICLA"] if \
        #                ind-syncLA[0] > firstPN-syncimu[0] \
        #                and ind-syncLA[0] < lastPN-syncimu[0]]

        #cycledtaold2 = [(start_-syncLA[0]+syncimu[0], stop_-syncLA[0]+syncimu[0]) \
        #                for (start_, stop_) in tr.attrs["PNAtCycleEvents"] \
        #                if start_-syncLA[0] > firstPN-syncimu[0] \
        #                and stop_-syncLA[0] < lastPN-syncimu[0]]

        cycledta = [(start_-syncLA[0]+syncimu[0], stop_-syncLA[0]+syncimu[0]) \
                for (start_, stop_) in tr.attrs["PNAtCycleEvents"] \
                if start_>firstPN and stop_<lastPN]

        return cycledta



    def get_imu_data(self, subject="S7", trial="D", imu="LH",\
                         startTime=60,\
                         anTime=120, rawData=nvg_2012_09_data,
                         split=False, SIUnits=False):
        """ Returns data for the specified subject, trial and imu.
        The data returned starts at the specified time into the trial, and has the
        specified length in seconds. The start of the trial is according to the
        time of the LA imu, since this is used for the cycle events. Because of
        slight out-of-sync start of the IMU data (the trial data for the different
        IMUs stored in the hdf file are NOT syncronized).

        New 2017-03-30: If startTime is negative, look up raw data file and
        read from this.

        Arguments:
        subject, trial, imu  -> self evident
        startTime, anTime    -> In seconds time out in trial to fetch data and
                                how many seconds to get. Can be negative or None
                                If None, data is read from the beginning of the
                                imu data file no matter the trial.
        rawData              -> callable that will return list of files
        split                -> If True, splits the data into gait cycles
                                and returns data in the form of a list
        SIUnits              -> If True, returns data where
                                imudt[:,-1]   - times in seconds, so the first
                                                column with packet numbers are kept
                                imudt[:,1:4]  - gyro data in rad/s
                                imudt[:,4:7]  - acc data in m/s^2


        Returns:
        imudta  <-  array with IMU data
        tr      <-  the trial dataset
        subj    <-  the subject dataset
        """

        syncimu = self.get_PN_at_sync(subject,imu)
        syncLA = self.get_PN_at_sync(subject,"LA")

        subj = self.hdfFile[subject]
        tr = subj[trial]
        imudta = tr[imu]

        imudtaLA = tr["LA"]
        startPN_LA = imudtaLA[0,0]
        startPN = imudta[0,0]

        PNsinceSyncLA = startPN_LA - syncLA[0]
        PNsinceSync = startPN - syncimu[0]
        PNoffset = PNsinceSyncLA - PNsinceSync
        #print PNoffset
        # If the offset is positive the imu data starts before the LA, and so
        # we should return data further ahead in the data array.

        sfreq = self.hdfFile.attrs['packetNumbersPerSecond']

        if startTime is not None:
            startTime += float(PNoffset)/float(sfreq)
        # Only every second packet present in data
        imudtPart = None
        if startTime >= 0:
            startInd = int(startTime*sfreq/2)
            endInd = int((startTime + anTime)*sfreq/2)
            # There could be a few packets missing, but we ignore this at the moment.
            imudtPart = np.array(imudta[startInd:endInd, :])
        elif startTime is None:
            # Read from start of data file
            startPacket = 0
            endPacket = anTime*sfreq
            (dta, events) = rawData()
            fnames = dta[subject][imu] # (DateTime.csv, CalInertialAndMag.csv)
            imudtPart = read_packets(fnames[1], startPacket, endPacket)
        else:
            startPacket = startTime*sfreq + imudta[0,0]
            endPacket = (startTime+anTime)*sfreq + imudta[0,0]
            (dta, events) = rawData()
            fnames = dta[subject][imu] # (DateTime.csv, CalInertialAndMag.csv)
            imudtPart = read_packets(fnames[1], startPacket, endPacket)

        if SIUnits:
            sfreq = self.hdfFile.attrs['packetNumbersPerSecond']
            dt = 1.0 / float(sfreq)
            tvec = imudtPart[:,0:1]*dt
            imudtPart[:,1:4] *= np.pi/180.0
            imudtPart[:,4:7] *= 9.82

            imudtPart = np.hstack( (imudtPart, tvec) )
        if split and startTime is not None:
            (imudtLA, subj_, trial_) = self.get_imu_data(subject, trial, "LA",
                                                            startTime, anTime)
            firstPN = imudtLA[0,0]
            lastPN = imudtLA[-1,0]
            #cycledtaNoShift= self.get_cycle_data(subject, trial, "LA",
            #                                                firstPN, lastPN)
            cycledtaNoShift= self.get_cycle_data(subject, trial, imu,
                                                            firstPN, lastPN)
            packetNumbers = np.asarray(imudtPart[:,0], np.int32)

            # Remove cycles outside range of data
            PNmax = packetNumbers[-1]
            PNmin = packetNumbers[0]

            cycledtaFixed = [ (start_, end_) for (start_, end_) in cycledtaNoShift
                                    if start_ > PNmin and end_ < PNmax]

            cycledtaInds = [ (np.where(packetNumbers <= cd_[0])[0][-1],
                              np.where(packetNumbers >= cd_[1])[0][0])
                                for cd_ in cycledtaFixed]
            imudtaSplit = [ imudtPart[startInd_:stopInd_, :]
                            for (startInd_, stopInd_) in cycledtaInds ]

            return (imudtaSplit, tr, subj)

        else:
            return (imudtPart, tr, subj)

    def estimate_joint_axes(self, subj, trial, imus, startTime=60, anTime=120,
                            stride=100):
        """
        Implementation of Seel's method to determine the joint axis of the
        joint connecting two segments.
        See Seel et al http://www.mdpi.com/1424-8220/14/4/6891/htm

        Returns:
            (axis0, axis1) <- a tuple with the two joint axes in the local
                              coordinate system of each imu
        """

        [imudt0, s_, t_] = self.get_imu_data(subj, trial, imus[0], startTime, anTime)
        g0 = imudt0[::stride,1:4]*np.pi/180.0
        [imudt1, s_, t_] = self.get_imu_data(subj, trial, imus[1], startTime, anTime)
        g1 = imudt1[::stride,1:4]*np.pi/180.0

        # Initial guesses: axis aliged with local z-axis
        x0 = np.zeros(4)

        def sph2vec(th,phi):
            return np.array([np.sin(th)*np.cos(phi),
                             np.sin(th)*np.sin(phi),
                             np.cos(th)])

        def residuals(x, g0, g1):
            j0 = sph2vec(x[0], x[1])
            j1 = sph2vec(x[2], x[3])

            N = len(g0)
            return (np.sum(np.cross(g0, np.tile(j0, (N,1)))**2, axis=1)
                    - np.sum(np.cross(g1, np.tile(j1, (N,1)))**2, axis=1))

        axes,cov,infodict,mesg,ier = optimize.leastsq(residuals,
                                            x0,
                                            args = (g0,g1),
                                            full_output=True)

        return (sph2vec(axes[0], axes[1]), sph2vec(axes[2], axes[3]) )

    def estimate_joint_center(self, subj, trial, imus, startTime=60, anTime=120,
                                stride=100):
        """
        Implementation of Seel's method to determine the center of the joint
        of two connected segments.
        See Seel et al http://www.mdpi.com/1424-8220/14/4/6891/htm

        Returns:
            (r0, r1) <- a tuple with the vector in the local coordinate system
            from the center of the IMU to the joint center of each IMU,
            respectively
        """

        sfreq = self.hdfFile.attrs['packetNumbersPerSecond']
        dt = 1.0/sfreq
        [imudt0, s_, t_] = self.get_imu_data(subj, trial, imus[0], startTime, anTime)
        tvec0 = imudt0[:,0]*dt
        g0 = imudt0[:,1:4]*np.pi/180.0
        a0 = imudt0[:,4:7]*9.82
        angacc0 = cpkinematics.ang_acc(g0, tvec0)
        #angacc0 = cpkinematics.ang_acc(g0, dt*2)

        [imudt1, s_, t_] = self.get_imu_data(subj, trial, imus[1], startTime, anTime)
        tvec1 = imudt1[:,0]*dt
        g1 = imudt1[:,1:4]*np.pi/180.0
        a1 = imudt1[:,4:7]*9.82
        angacc1 = cpkinematics.ang_acc(g1, tvec1)
        #angacc1 = cpkinematics.ang_acc(g1, dt*2)

        r00 = np.array([0.1,0,0.])
        r10 = np.array([-0.1,0,0.])


        # If using residuals2, compute omega matrices
        Omega0 = cpkinematics.omega_matrix(g0, angacc0)
        Omega1 = cpkinematics.omega_matrix(g1, angacc1)

        def residuals2(x, a0, Omega0, a1, Omega1):
            r0 = x[:3]
            r1 = x[3:]


            jacc0 = np.array([a0[i] + np.dot(Omega0[...,i], r0)
                        for i in range(Omega0.shape[-1]) ])
            jacc1 = np.array([a1[i] + np.dot(Omega1[...,i], r1)
                        for i in range(Omega1.shape[-1]) ])

            return (np.sqrt(np.sum(jacc0**2, axis=1))
                            - np.sqrt(np.sum(jacc1**2, axis=1)) )

        def residuals(x, a0, g0, angacc0, a1, g1, angacc1):
            r0 = x[:3]
            r1 = x[3:]

            jacc0 = cpkinematics.joint_center_acceleration(a0, g0, r0, [], angacc0)
            jacc1 = cpkinematics.joint_center_acceleration(a1, g1, r1, [], angacc1)

            #return (np.sum(jacc0**2, axis=1) - np.sum(jacc1**2, axis=1))
            return (np.linalg.norm(jacc0, axis=1)
                            - np.linalg.norm(jacc1, axis=1))

        #print residuals(np.hstack((r00, r10)), a0, g0, angacc0,
        #                    a1, g1, angacc1)[:20]
        #print residuals2(np.hstack((r00, r10)), a0, Omega0,
    #                        a1, Omega1)[:20]

        rr,cov,infodict,mesg,ier = optimize.leastsq(residuals,
                                            np.hstack((r00, r10)),
                                            args = (a0,g0,angacc0,a1,g1,angacc1),
                                            full_output=True)
        #rr,cov,infodict,mesg,ier = optimize.leastsq(residuals2,
        #                            np.hstack((r00, r10)),
        #                            args = (a0,Omega0,a1,Omega1),
        #                            full_output=True)

        return (rr[:3], rr[3:] )

    def estimate_joint_center_matlab(self, subj, trial, imus, startTime=60, anTime=120,
                                stride=100):
        """
        Implementation of Seel's method to determine the center of the joint
        of two connected segments.
        See Seel et al http://www.mdpi.com/1424-8220/14/4/6891/htm

        Uses Fredrik Olssons matlab code

        Returns:
            (r0, r1) <- a tuple with the vector in the local coordinate system
            from the center of the IMU to the joint center of each IMU,
            respectively
        """

        sfreq = self.hdfFile.attrs['packetNumbersPerSecond']
        dt = 1.0/sfreq
        [imudt0, s_, t_] = self.get_imu_data(subj, trial, imus[0], startTime, anTime)
        tvec0 = imudt0[:,0]*dt
        g0 = imudt0[:,1:4]*np.pi/180.0
        a0 = imudt0[:,4:7]*9.82
        angacc0 = cpkinematics.ang_acc(g0, tvec0)
        #angacc0 = cpkinematics.ang_acc(g0, dt*2)

        [imudt1, s_, t_] = self.get_imu_data(subj, trial, imus[1], startTime, anTime)
        tvec1 = imudt1[:,0]*dt
        g1 = imudt1[:,1:4]*np.pi/180.0
        a1 = imudt1[:,4:7]*9.82
        angacc1 = cpkinematics.ang_acc(g1, tvec1)
        #angacc1 = cpkinematics.ang_acc(g1, dt*2)

        return fomatlab.estimate_joint_center(a0, g0, angacc0, a1, g1, angacc1, None)

    def add_imu_data(self, subgroup, datafiles, eventLogFile, syncdata=None,initTime=240):
        """ Adds imu data for a single trial for one subject. Deletes any data
        for this subject already in the file.
        The datafiles are synced using the sync event
        (a sharp negative acc in x-direction) occurring in the beginning
        of the experiment.
        :param subgroup: The subgroup whose data should be addded. Example ``S3``
        :type subgroup: Str
        :param datafiles: Dictionary providing full path to csv data files.
        Expected keys ``["LA", "LT", "LH", "B", "N", "RA", "RT", "RH"]``
        :type datafiles: Dict
        :param eventLogFile: Timestamps and keystrokes in chronological order
        :type eventLogFile: Str
        :param syncdata: If given, is list of packetnumbers at sync event
        :type syncdata: List
        :param initTime: Time interval at beggining of each file containing the sync
        pulse.
        :type initTime: int seconds
        :raises: ValueError, if not able to read and handle data
        """

        if syncdata == None:
            # First sync data
            deltaT = timedelta(seconds=initTime)
            deltaT0 = timedelta(seconds=5)
            initdata = []
            timeKeepers = []
            for key, fnames in datafiles.iteritems():
                if len(fnames) == 2:
                    timeKeeper = TimeKeeper(fnames[0])
                    time0 = timeKeeper.first_timestamp()
                    initdata.append(read_part(fnames[1], time0+deltaT0, time0+deltaT, timeKeeper))
                    timeKeepers.append(timeKeeper)

            (packetN,sign) = sync_signals(initdata, timeKeepers)
            #plot_sync(initdata.values(), packetN)
        else:
            packetN = syncdata

        # Store the packet number at sync
        f = self.hdfFile
        g = None
        try:
            g = f.create_group(subgroup)
        except ValueError:
            # Subject data exists. Delete first
            del f[subgroup]
            g = f.create_group(subgroup)

        g.attrs["PNAtSync"] = packetN

        # Read in events
        events = read_event_log(eventLogFile)

        # Correct for missing events for S2
        if subgroup == "S2":
            events.insert(0, [datetime(2012,9,18,13,51), 'd'])
            events.insert(0, [datetime(2012,9,18,13,41), 'd'])

        # Go through the key events, and store data for each trial
        trials = ['N', 'D', 'B', 'M']
        for tr in trials:
            # The parsing of the event sequence is a bit complicated since
            # we need to check if an event is cancelled by three consecutive
            # 'c' keys, or if two events are close, then the last one is
            # the one to use
            ind = 0
            trialStart = None
            trialEnd = None
            while ind < len(events):
                key = events[ind][1]
                if key.upper() == tr:
                    # Check if the event was cancelled
                    if ind+3 < len(events):
                        if (events[ind+1][1].upper() == 'C' and \
                                events[ind+2][1].upper() == 'C' and \
                                events[ind+3][1].upper() == 'C'):
                            ind += 3
                            continue
                    eventTime = events[ind][0]
                    if trialStart == None:
                        trialStart = eventTime
                    else:
                       # Check if the event occurred shortly after the previous
                        if eventTime-trialStart < timedelta(minutes=3):
                            trialStart = eventTime
                        else:
                            if trialEnd == None:
                                trialEnd = eventTime
                            else:
                                # Check if the event occurred shortly after the previous
                                if eventTime-trialEnd  < timedelta(minutes=3):
                                    trialEnd = eventTime
                ind += 1

            # If trialStart exists, but not trialEnd, then use 10 minutes from trialStart
            if (trialStart != None and trialEnd == None):
                trialEnd = trialStart + timedelta(minutes=10)

            if (trialEnd == None or trialStart == None):
                raise ValueError("Unable to find start and end events")

            trialGroup = None
            try:
                trialGroup = g.create_group(tr)
            except ValueError:
                # Created, so delete first
                del g[tr]
                trialGroup = g.create_group(tr)

            # Now read csv data
            for key, fnames in datafiles.iteritems():
                if len(fnames) == 2:
                    timeKeeper = TimeKeeper(fnames[0])
                    imuData = read_part(fnames[1], trialStart, trialEnd, timeKeeper)
  #                  try:
                    print "Adding dataset %s in trial %s for subject %s"\
                        % (key, tr, subgroup)
                    trialGroup.create_dataset(key, data=imuData)
   #                 except ValueError:
    #                    print "Unexpected exception. Data set %s should not exist in group %s" %(key, tr)
            self.hdfFile.flush()

    def get_PN_at_sync(self, subject, imu):
        """
        Returns a tuple containing the packet number at the sync pulse for the imu
        and the row-index in the data file of the pulse (the latter not so useful)
        """
        subj = self.hdfFile[subject]
        PNs = subj.attrs['PNAtSync']
        if PNs.shape[0] == 7:
            # No B data
            keys = ["RT", "LA", "LH", "N", "LT", "RA", "RH"]
        else:
            keys = ["RT", "B", "LA", "LH", "N", "LT", "RA", "RH"]

        ind = keys.index(imu)
        return PNs[ind]


    def check_sync(self, subject, figsize = (10,18), dtaLength=60):
        """
        Loads the initial part of the IMU data files, and plots together with the
        identified syncpulse for each imu. Used to check if there are any errors in
        PNAtSync"
        """

        pyplot.figure(figsize=figsize)
        k_ = 1 # counter
        for imu in self.hdfFile[subject]["N"].iterkeys(): # Doesn't matter which trialdata
            (imudt,t_, s_) = self.get_imu_data(subject, "N", imu,
                                                startTime=None,
                                                anTime=dtaLength)
            acc = np.mean(imudt[:,4:7]**2, axis=1)
            pyplot.subplot(4,2,k_)
            pyplot.plot(imudt[:,0], acc )
            syncPN = self.get_PN_at_sync(subject, imu)
            pyplot.plot([syncPN[0], syncPN[0]], [-5,15])
            pyplot.title(imu)
            k_ += 1
        pyplot.show()

    def get_comparison_data(self, mocapdatafile, markers,  subject, trial, imus,
                            startTime=60, anTime=180, plotResults=False):
        """ Returns imu data and corresponding mocapdata

        Arguments:
        mocapdatafile -> file name with full path
        markers       -> list of marker names, e.g. ['ANKLE', ̈́"KNEE", 'THIGH']
        subject       -> Subject. String such as 'S4'
        trial         -> Trial. Can be either 'D' or 'N'
        imus          -> List of imus, e.g ['LA', 'LT']
        startTime     -> Time after start of trial to read data
        anTime        -> Number of seconds of data to return

        Returns tuple:
        md    <-  dict with markerdata
        imudt <-  imu data
        """

        # Load imu data. These will be synced
        imudata = {}
        for imu in imus:
            (imudt, subj_, trial_) = self.get_imu_data(subject, trial, imu,
                                                                startTime, anTime)
            imudata[imu] =  imudt


        (imudt, subj_, trial_) = self.get_imu_data(subject, trial, "LA",
                                                                startTime, anTime)
        firstPN = imudt[0,0]
        lastPN = imudt[-1,0]
        syncLA = self.get_PN_at_sync(subject, "LA")
        dt = 1.0/262.0 # Weird, but this is the actual sampling period
        tvec = dt*(imudt[:,0]-syncLA[0])
        cycledtaNoShift= self.get_cycle_data(subject, trial, "LA", firstPN, lastPN)

        packetNumbers = np.asarray(imudt[:,0], np.int32)
        cycledtaInds = [ (np.where(packetNumbers <= cd_[0])[0][-1],
                          np.where(packetNumbers >= cd_[1])[0][0])
                            for cd_ in cycledtaNoShift ]

        cycledtaSec = [ ( (cd_[0]-syncLA[0])*dt, (cd_[1]-syncLA[0])*dt )
                            for cd_ in cycledtaNoShift]

        imudata['cycledata'] = cycledtaNoShift # PNs of LA imu at events
        imudata['cycledataSec'] = cycledtaSec # times in seconds since sync
        imudata['cycledataInds'] = cycledtaInds # times in seconds since sync

        return (imudata, self.get_marker_data(subject, trial, markers,
                                startTime, anTime, rawData))

    def get_marker_data(self, subject="S7", trial="D",
                            markers=["ANKLE", "WRIST"],
                            startTime=5*60, anTime=120, rawData=nvg_2012_09_data,
                            split=False):
        """ Returns marker data for the specified subject and trial.
        The data returned starts at the specified time into the trial, and has the
        specified length in seconds. The start of the trial is according to the
        time of the LA imu, since this is used in the definition of the start
        of cycles.

        Arguments:
        subject, trial, markers  -> self evident
        startTime, anTime    -> In seconds time out in trial to fetch data and
                                how many seconds to get.
        split                -> If True, splits the data into gait cycles
                                and returns data in the form of a list

        Returns:
        markerdta  <-  array with IMU data
        """

        if not trial in ("N", "D"):
            raise ValueError("Marker data only available for trial D or N")

        subjdta, eventdta = rawData()

        try:
            markerfile = subjdta[subject]["MD-" + trial]
        except LookupError:
            raise ValueError("Markerdata not found for subject " + subject)

        md = qtsv.loadQualisysTSVFile(markerfile)

        timeSinceSync = md.timeStamp - md.syncTime

        (imudt, subj_, trial_) = self.get_imu_data(subject, trial, "LA",
                                                                startTime, anTime)
        firstPN = imudt[0,0]
        lastPN = imudt[-1,0]
        syncLA = self.get_PN_at_sync(subject, "LA")
        dt = 1.0/262.0 # Weird, but this is the actual sampling period
        firstPacketTimeSinceSync = (firstPN - syncLA[0])*dt
        lastPacketTimeSinceSync = (lastPN - syncLA[0])*dt

        # OBS: md.frameTimes start at zero
        frames2use = md.frameTimes[md.frameTimes > (firstPacketTimeSinceSync
            - timeSinceSync.total_seconds())]
        frames2use = frames2use[frames2use < (lastPacketTimeSinceSync
            - timeSinceSync.total_seconds())]

        ft = frames2use + timeSinceSync.total_seconds()

        #1/0
        mdata = {'frames':frames2use}
        mdata['frametimes'] = ft
        if "ANKLE" not in markers:
            markers.append("ANKLE")
        for m in markers:
            mdata[m] = md.marker(m).position(frames2use).T


        (cyclesTime, cyclesInd) = markerdata.get_marker_data_cycles(md,
                                            frames2use, plotResults=self.debug)

        mdata['cycledataTime'] = cyclesTime
        mdata['cycledataInd'] = cyclesInd

        return mdata


    def set_standing_reference(self):
        """"
        Goes through all subjects, plots IMU data before start of N trial.
        Lets user pick start and end of interval where the subject is standing
        still. Stores values of imu data at this interval in the attribute
        hdfFile[subj].attrs['standingReferenceIMUdata']
        """

        sref = self.apply_to_all_trials(self._pick_standing_reference,
                                                                triallist=["N"])

    def _pick_standing_reference(self, subj, trial):
        """
        Loads data from two minutes before start of trial. Plots and lets the
        user choose an interval for standing reference. The interval is
        adjusted to 200 frames (about 1.5s) and returned
        """

        LAdta, tr_, s_ = self.get_imu_data(subj, trial, 'LT', startTime=-60, anTime=120)
        RAdta, tr_, s_ = self.get_imu_data(subj, trial, 'RT', startTime=-60, anTime=120)

        pyplot.figure()
        pyplot.subplot(221)
        pyplot.plot(LAdta[:,1:4])
        pyplot.title("LT gyro")
        pyplot.subplot(222)
        pyplot.plot(LAdta[:,4:7])
        pyplot.title("LT acc")
        pyplot.subplot(223)
        pyplot.plot(RAdta[:,1:4])
        pyplot.title("RT gyro")
        pyplot.subplot(224)
        pyplot.plot(RAdta[:,4:7])
        pyplot.title("RT acc")

        intervalSet = False
        while not intervalSet:
            pyplot.show()
            interval = raw_input("Set range [startindex, stopindex]: ")
            try:
                (startstr, stopstr) = interval.split(",")
                start_ = int(float(startstr))
                stop_ = int(float(stopstr))
                intervalSet = True
            except ValueError:
                pass

        stop = min(stop_, start_+200)

        subjdta = self.hdfFile[subj]

        srefGroup = None
        sr = "standingReference"
        try:
            srefGroup = subjdta.create_group(sr)
        except ValueError:
            # Group already created, so delete first
            del subjdta[sr]
            srefGroup = subjdta.create_group(sr)

        for imu_ in subjdta[trial]:
            print "Adding standing reference for imu %s for subject %s" % (imu_, subj)
            try:
                imudta, s_, t_ = self.get_imu_data(subj, trial, imu_,
                                                    startTime=-60, anTime=120)
                srefGroup.create_dataset(imu_,
                                        data=np.asarray(imudta[start_:stop_, :]) )
            except IOError:
                # Cannot find file
                print "Failed to set standing reference for imu %s" %(imu_,)

            self.hdfFile.flush()

        self.hdfFile.flush()

    def get_reference_imu_data(self, subject, imu, SIUnits=True):
        """
        Returns time-averages of the gyro, acc and mag data from the reference
        measurement (standing still)
        """
        gyro, acc, mag = self.get_raw_reference_imu_data(subject, imu, SIUnits)
        return (np.mean(gyro, axis=0),
                    np.mean(acc, axis=0),
                    np.mean(mag, axis=0) )

    def get_raw_reference_imu_data(self, subject, imu, SIUnits=True):
        """
        Returns time-averages of the gyro, acc and mag data from the reference
        measurement (standing still)
        """
        srefdta = self.hdfFile[subject]["standingReference"][imu]
        gyro = np.array(srefdta[:,1:4])
        acc = np.array(srefdta[:,4:7])
        mag = srefdta[:,7:10] # Not copying since this will never be altered.

        if SIUnits:
            gyro *= np.pi/180.0
            acc *= 9.82

        return (gyro, acc, mag)

    def plot_reference_imu_data(self, subject, imu, SIUnits=False):

        gyro, acc, mag = self.get_raw_reference_imu_data(subject, imu, SIUnits)

        if SIUnits:
            gyrolabel = "Gyro [rad/s]"
            acclabel = "Acc [m/s^2]"
        else:
            gyrolabel = "Gyro [deg/s]"
            acclabel = "Acc [g]"


        pyplot.figure(figsize=(12,10))

        pyplot.subplot(3,1,1)
        pyplot.plot(gyro)
        pyplot.ylabel(gyrolabel)

        pyplot.subplot(3,1,2)
        pyplot.plot(acc)
        pyplot.ylabel(acclabel)

        pyplot.subplot(3,1,3)
        pyplot.plot(mag)
        pyplot.ylabel("Magn [G]")
        pyplot.legend(("x", "y", "z"))

        pyplot.title("IMU data at reference measurement for subject %s, trial %s, imu %s" % (subject, trial, imu))
        pyplot.show()

    def get_reference_vertical(self, subject, imu):
        """
        Returns a unit vector in the vertical direction taken from the standing
        reference imu data
        """
        srefdta = self.hdfFile[subject]["standingReference"][imu]
        acc = np.mean(srefdta[:,4:7], axis=0)
        return acc/np.linalg.norm(acc)

    def _integrate_acc(self, acc):
        """ Integrate acceleration to get velocity and displacement. The
            integration is done using the trapezoidal rule, and reset at
            the start of each cycle. The mean acceleration and mean displacement
            are removed during each cycle.
        """
        [v, g] = self._integrate_cyclic_trapezoidal(acc)
        [d, slask] = self._integrate_cyclic_trapezoidal(v)

        return [d, v, g]


    def _integrate_cyclic_trapezoidal(self, a):

        nfrs = a.shape[0]
        v = np.zeros((nfrs, 4))
        v[:,0] = a[:,0]
        currentV = np.array([0.0, 0.0, 0.0])
        sfreq = self.hdfFile.attrs['packetNumbersPerSecond']
        h = 1.0/sfreq
        lastT = a[0,0]*h
        for i in range(1,nfrs):
            t = a[i,0]*h
            dt = t-lastT
            lastT = t
            currentV += dt*0.5*(a[i-1, 1:4] + a[i, 1:4])
            v[i,1:4] = currentV

        #tau = (a[:,0]- a[0,0])*h
        g = np.mean(a[:,1:], axis=0)
        #v[:, 1:4] -= np.outer(tau,g)
        return [v, g]


class TimeKeeper:
    '''
    Reads table of packet number and time stamp generated by x-imu.
    The order of the columns is
    Packet number, year, month, day, hours, minutes, seconds
    '''
    def __init__(self, csvfilename):
        csvfile = open(csvfilename, 'r')
        reader = csv.reader(csvfile, delimiter=',')
        self.headings = reader.next()
        self.ttable = [[int(row[0]), \
                            datetime(int(row[1]), int(row[2]), int(row[3]), \
                                         int(row[4]), int(row[5]), int(row[6]))]\
                           for row in reader]

        # Choosing the distance between the last and first row to compute sampling period
        #
        dt = self.ttable[-1][1] - self.ttable[0][1]
        dp = self.ttable[-1][0] - self.ttable[0][0]

        self.dt = dt / dp

    def get_packet(self, timestamp):
        '''
        If the exact timestamp is found, then the corresponding packet number is returned.
        Otherwise: Finds the first table entry before given timestamp and the last after.
        Then interpolates between.
        '''
        packet = [packettime[0] for packettime in self.ttable if (packettime[1] == timestamp)]
        if packet != []:
            return packet[0]
        else:
            before = [packettime for packettime in self.ttable if (packettime[1] < timestamp)]
            after = [packettime for packettime in self.ttable if (packettime[1] > timestamp)]
            rowbefore = before[-1]

            if len(after) == 0:
                # Asked to access packet after last timestamp. Just return the last packet
                return  self.ttable[-1][0]

            rowafter = after[0]
            dt = rowafter[1] - rowbefore[1] # The gap in time between the rows in the table
            dp = rowafter[0] - rowbefore[0]
            dt1 = timestamp - rowbefore[1]
            # Not needed dt2 = rowafter[1] - timestamp
            s = float(dt1.seconds*10**3 + dt1.microseconds/1000) / float(dt.seconds*10**3 + dt.microseconds/1000)
            return (rowbefore[0] + int(s*dp))-1

    def get_time(self, packet):
        '''
        If the exact packet number is found, then the corresponding timestamp is returned.
        Otherwise linear interpolation
        '''
        time = [packettime[1] for packettime in self.ttable if (packettime[0] == packet)]
        if packet != []:
            return packet[0]
        else:
            before = [packettime for packettime in self.ttable if (packettime[0] < packet)]
            rowbefore = before[-1]
            dp1 = int(packet - rowbefore[0])
            return (rowbefore[1] + dp1*self.dt)

    def first_timestamp(self):
        return self.ttable[0][1]

def find_narrow_peaks(signal_, threshold, peakwidth, mindist=100, maxdist=400):
    """ Will look for narrow peaks in the signal which are above the given threshold
    and of width less than peakwidth.
    Returns a list of the first indices in each narrow peak
    """
    indAbove = np.nonzero(signal_ > threshold) # Returns a two-tuple
    indAbove = indAbove[0]
    # Must be able to handle peaks of one sample width, so add also the next index
    # to all indices above the threshold
    indAbove = np.unique(np.concatenate((indAbove, indAbove+1)))
    indAdiff = indAbove[1:] - indAbove[:-1]
    indConseq = indAdiff == 1 # booelan array
    narrow_peaks = []
    ind = 0
    peak = -1
    currentwidth = 0
    while ind < len(indConseq):
        if not indConseq[ind]:
            # Check if we were considering a peak. If so, then it is a
            # legible narrow peak, and should be added to the list
            if peak > -1:
                narrow_peaks.append(peak)

            peak = -1
            ind += 1
            continue
        # So, indConseq[ind] is true. If not already considering a peak,
        # do so
        if peak < 0:
            peak = indAbove[ind]
            currentwidth = 1
        else:
            # Already considering this as a peak. Increment the peakwidth, and
            # see if it is above the maxwidth
            currentwidth += 1
            if currentwidth > peakwidth:
                #print "DEBUG ind=%d, currentwidth=%d, peakwidth=%d" \
                #    % (ind,currentwidth, peakwidth)
                peak = -1
                ind += 1
                while ind < len(indConseq):
                    if not indConseq[ind]:
                        break
                    ind += 1
        ind += 1
    # Now that the narrow peaks are found, look for a double peak at least mindist apart,
    # but not more than maxdist.


    # Problem with the code below. Return just all peaks
    return narrow_peaks

    if len(narrow_peaks) < 2:
        return []

    firstpeak = narrow_peaks[0]
    for p in narrow_peaks[1:]:
        peakdistance = p-firstpeak
        if peakdistance > mindist and peakdistance < maxdist:
            return [firstpeak,]
        else:
            firstpeak = p
    # If this line is reached, then no double peaks were found
    return []

def read_part(csvfilename, startTime, endTime, timeKeeper):
    '''
    Reads the csv file and returns the part of the file which was recorded
    between startTime and endTime
    '''
    startPacket = timeKeeper.get_packet(startTime)
    endPacket = timeKeeper.get_packet(endTime)

    return read_packets(csvfilename, startPacket, endPacket)

def read_packets(csvfilename, startPacket=None, endPacket=None):
    """
    Reads the imu data in the csv file between startPacket and endPacket.
    If startPacket is None, read from first row. If endPacket is None read
    to end of file.
    """
    csvfile = open(csvfilename, 'rb')
    reader = csv.reader(csvfile)
    reader.next() # Skip first row which contains headers

    if startPacket is not None:
        # Discard rows until startPacket found
        row = reader.next()
        while ( int(row[0]) < startPacket): row = reader.next()


    print(reader.next())

    if endPacket is not None:
        # Read rows until endPacket found
        return np.array([[float(col) for col in row]  for row in
                        list(itertools.takewhile(
                        lambda x: int(x[0]) <= endPacket, reader))])
    else:
        return np.array([[float(col) for col in row]  for row in reader])




def main():
    #import sys
    #if len(sys.argv) == 0:
    dtapath = "/home/kjartan/Dropbox/projekt/nvg/data/test0910/6522/"
    dtaset = "06522";
    dttme1 = TimeKeeper(dtapath + dtaset + "_DateTime.csv")
    return dttme1


def manually_set_range(signal, timekeeper, channel):

    tstart = timekeeper.get_time(signal[0,0])
    tend = timekeeper.get_time(signal[-1,0])

    dt = timedelta(microseconds=10.0**6/128)
    time = mdates.drange(tstart, tend, dt)

    #1/0
    #
    #    np.linspace(tstart.minute*60*10**6 + tstart.second*10**6 + tstart.microsecond,\
    #                    tend.minute*60*10**6 + tend.second*10**6 + tend.microsecond,\
    #                   signal.shape(0))
    pyplot.figure()
    pyplot.plot(time, signal[:len(time),channel])
    ax = pyplot.gca()

    # matplotlib date format object
    hfmt = mdates.DateFormatter('%M:%S')

    ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=range(0,60,10)))
    ax.xaxis.set_major_formatter(hfmt)
    pyplot.xticks(rotation='vertical')
    pf = pointfinder.PointFinder(time)
    pyplot.connect('button_press_event', pf)
    pyplot.show()
    while raw_input("Done setting range? [y/n]: ").lower() != "y":
        # Wait for user to confirm
        i = 0
    sel = pf.get_selected()

    intBefore = np.nonzero(time < min(sel))
    intAfter = np.nonzero(time > max(sel))
    ntime = time[intBefore[0][-1]:intAfter[0][0]]
    nsignal = signal[intBefore[0][-1]:intAfter[0][0], :]

    #intStart = np.nonzero(time == min(sel))
    #intEnd = np.nonzero(time == max(sel))
    #signal = signal[intStart[0][0]:intEnd[0][0], :]


    pyplot.clf()
    pyplot.plot(ntime, nsignal[:,4])
    ax = pyplot.gca()
    ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=range(0,60,4)))
    ax.xaxis.set_major_formatter(hfmt)
    pyplot.xticks(rotation='vertical')

    return nsignal




def sync_signals(signals_, timeKeepers, channel=4, threshold=0.4,
                 maxpeakwidth=40, checkConcensus=False):
    """ Will look for two narrow peaks more than 100 samples but less than
    400 samples apartwithin  in the list of signals at the given channel.
    The first threshold passing is taken as the sync signal.
    The packet number and signal index at the peak are returned in a list.

    """
    mindist = 100
    maxdist = 400
    mysignals = []
    for s in signals_:
        acc = s[:, channel:channel+3].copy()
        acc[:,0] = acc[:,0] + 1
        accmagn = np.sqrt(np.sum(acc**2, axis=1))
        s_ = np.column_stack([s[:,0], accmagn])
        mysignals.append(s_)

    concensus = False
    attempt = 0
    while (not concensus and attempt < 10):
        attempt += 1
        packetNumbers = []
        timeStamps = []
        for [s, tk] in itertools.izip(mysignals, timeKeepers):
            indxs = []
            thr = threshold
            while len(indxs) == 0 and thr > 0 :
                indxs = find_narrow_peaks(s[:,1], thr, maxpeakwidth, mindist, maxdist)
                loggin.DEBUG("Found %d peaks using threshold %1.3f" % (len(indxs), thr))
                thr -= threshold/100.0
            indx = indxs[0]
            packetNumbers.append([s[indx,0], indx])
            synctime = tk.get_time(s[indx,0])
            timeStamps.append(synctime.minute*60+synctime.second)

        if checkConcensus:
            # Test for concensus about the time of the sync signal
            ts = np.array(timeStamps)
            tsdev = np.abs(ts - np.median(ts))
            outliers = np.nonzero(tsdev > 5)
            for ol in outliers[0]:
                logging.DEBUG("Deviating time of sync in signal %d: %d seconds"% (ol, tsdev[ol]))
                pn = packetNumbers[ol]
                elindx = range(pn[1]-10, pn[1]+10)
                s = mysignals[ol]
                s[np.ix_(elindx, [1])] = 0
                if len(outliers[0]) == 0:
                    concensus = True
        else:
            concensus = True
    if not concensus:
        logging.warning("Could not find timestamps to agree at sync pulse \
            Could be due to wrong setting of time on imu.")

    return [packetNumbers, mysignals]

def plot_sync(signals_, packetAtSync, imunames, channel=4):
    pyplot.figure()
    for [s, sp, imu] in itertools.izip(signals_, packetAtSync, imunames):
        spn = sp[0] # The packet number
        spi = sp[1] # The corresponding row index
        #rowindx = range(max(spi-2000, 0), min(spi+2000, s.shape[0]))
        rowindx = range(0, min(spi+2000, s.shape[0]))
        colindx = range(channel, channel+3)
        t = s[np.ix_(rowindx, [0])] - spn
        if s.shape[1] == 2:
            accmagn = s[np.ix_(rowindx,[1])]
        else:
            2/0
            acc = [np.ix_(rowindx, colindx)].copy()
            acc[:,0] = acc[:,0] + 1
            accmagn = np.sqrt(np.sum(acc**2, axis=1 ))

        pyplot.plot(t, accmagn)

        if accmagn[0:10].mean() > 0.2:
            logging.warning("Unexpected large acceleration (%1.4f) at beginning for imu %s " \
                % (accmagn[0:10].mean(), imu) )

        if accmagn[-10:].mean() > 0.2:
            logging.warning("Unexpected large acceleration (%1.4f) at end for imu %s " \
                % (accmagn[-10:].mean(), imu) )


        pyplot.show()

def _sync_signals_old(signals_, timeKeepers, channel=4, threshold=1.0,\
                          checkConsensus=False):
    """ Will look for a peak in the list of signals at the given channel.
    The packet number at the peak is returned in a list.

    """

    find_ = np.argmax

    mysignals = []
    for s in signals_:
        mysignals.append((s+1.0)**2)

    concensus = False
    attempt = 0
    while (not concensus and attempt < 10):
        packetNumbers = []
        timeStamps = []
        for [s, tk] in itertools.izip(mysignals, timeKeepers):
            indx = find_(s[:,channel], axis=0)
            packetNumbers.append([s[indx,0], indx])
            synctime = tk.get_time(s[indx,0])
            timeStamps.append(synctime.minute*60+synctime.second)

        if checkConcensus:
            # Test for concensus about the time of the sync signal
            ts = np.array(timeStamps)
            tsdev = np.abs(ts - ts.mean())
            outliers = np.nonzero(tsdev > 10)
            for ol in outliers[0]:
                print "Deviating time of sync in signal %d: %d seconds"% (ol, tsdev[ol])
                pn = packetNumbers[ol]
                elindx = range(pn[1]-10, pn[1]+10)
                s[np.ix_(elindx, [channel])] = 0
                if len(outliers[0]) == 0:
                    concensus = True
        else:
            concensus = True
    if not concensus:
        print "Could not find timestamps to agree at sync pulse \
            Could be due to wrong setting of time on imu."

    return packetNumbers

def add_to_event_log(theDB, timestamp, key):
    """ Opens the event log and adds the timestamp and key pair """
    try:
        conn = sqlite3.connect(theDB, detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        c.execute('INSERT INTO testlog (timestamp, note) values (?,?)', (timestamp, key))
        conn.commit()
        c.close()
    except sqlite3.OperationalError, msg:
        print 'Sqlite error: ' + msg.message
def read_event_log(theDB):
    """ Reads keystroke events from the sqlite database provided.
    Returns the keys and timestamps in a list
    """
    try:
        conn = sqlite3.connect(theDB)
        c = conn.cursor()

        # select rows of interest from database
        events = []
        for row in c.execute("SELECT * FROM testlog WHERE note IN ('b','m','n','d','c')"):
            try:
                events.append([datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f"), row[1]])
            except ValueError:
                events.append([datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"), row[1]])

        c.close()

        # Sort in chronological order
        return sorted(events, key=lambda ev: ev[0])

    except sqlite3.OperationalError, msg:
        print 'Sqlite error: ' + msg.message

def create_event_log(theDB, events):
    """ Will create a new event log sqlite file with the events provided """
    try:
        conn = sqlite3.connect(theDB)
        c = conn.cursor()
        # Create a table to hold the event log
        c.execute('CREATE TABLE IF NOT EXISTS testlog (timestamp datetime, note nchar(1))')
        conn.commit()
        c.close()
    except sqlite3.OperationalError, msg:
        print 'Sqlite error: ' + msg.message

    for (timestamp, key) in events:
        add_to_event_log(theDB, timestamp, key)

def list_files():
    [dta, events] = nvg_2012_09_data()
    for [subject, subjdata] in dta.iteritems():
        print "==========================================="
        print "Order of IMU data for subject " + subject
        print "==========================================="
        for [imu, fnames] in subjdata.iteritems():
            print imu



def split_files_main(db, rawData=nvg_2012_09_data, initStart=20, initLength=120):
    """ This function reads in raw data, syncs across the different IMUs, splits it according to trial condition, and adds to the database. """
    [dta, events] = rawData()

    dT1 = timedelta(seconds=initStart)
    dT2 = dT1 + timedelta(seconds=initLength)

    for [subject, subjdata] in dta.iteritems():
        #if subject not in ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12" ]:
        if subject not in [ "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12" ]:
            print "==========================================="
            print "Adding data for subject " + subject
            print "==========================================="
            timers = []
            initdta = []
            imuname = []
            for [imu, fnames] in subjdata.iteritems():
                if len(fnames) == 2:
                    imuname.append(imu)
                    tk = TimeKeeper(fnames[0])
                    timers.append(tk)
                    t0 = tk.first_timestamp()
                    initdta.append(read_part(fnames[1], t0+dT1, t0+dT2, tk))
            # Manual handling of initial data before automatically finding sync signal
            myinitdata = []
            for [idta, tk] in itertools.izip(initdta, timers):
                print "Length of init data before setting range: %d" % (idta.shape[0],)
                nidta = manually_set_range(idta, tk, channel=4)
                print "Length of init data after setting range: %d" % (nidta.shape[0],)
                myinitdata.append(nidta)

            [packetNumberAtSync, syncdata] = sync_signals(myinitdata, timers)
            plot_sync(syncdata, packetNumberAtSync, imuname)

            answer = raw_input("Sync OK? [y/n]: ")
            if answer.lower() != "y":
                print "Skipping this subject. Fix sync."
                #2/0
                continue

            db.add_imu_data(subject, subjdata, events[subject], syncdata=packetNumberAtSync)


    # Plot data
    #two_plot.plotwndw.plotData(neckinit[:,0], neckinit[:,4:6])


    return [initdta, packetNumberAtSync]


#-------------------------------------------------------------------------------
# Unit tests
#-------------------------------------------------------------------------------

class TestXIMU(unittest.TestCase):

    def test_acc_tintegration(self):
        tEnd = 10
        sf = 256
        n = tEnd*sf
        w = 2*np.pi
        t = np.linspace(0,tEnd, n)
        d = np.sin(w*t)
        v = w*np.cos(w*t)
        g = 9.82;
        a = -w*w*np.sin(w*t) + g

        inds = np.arange(n)
        acc = np.column_stack((inds[::2], a[::2], np.zeros((n/2,2))))
        T = sf/w*2*np.pi
        cycledta = [i*T for i in range(10)]

        [di, vi] = _integrate_acc(acc,cycledta)

        pyplot.figure()
        pyplot.subplot(2,1,1)
        pyplot.plot(inds, d)
        pyplot.plot(di[:,0], di[:,1])
        for ind in cycledta:
            pyplot.plot([ind, ind], [-1, 1], 'm')

        pyplot.subplot(2,1,2)
        pyplot.plot(inds, v)
        pyplot.plot(vi[:,0], vi[:,1])
        for ind in cycledta:
            pyplot.plot([ind, ind], [-1, 1], 'm')

        pyplot.show()



    def test_find_narrow_peak(self):

        s = np.random.random((100,1))
        s[10,0] = 2
        s[20:25,0] = 3
        s[40:51,0] = 4

        # Should only find the middle peak
        narrowpeaks = find_narrow_peaks(s, threshold=2.9, peakwidth=6)
        print narrowpeaks

        self.assertTrue( len(narrowpeaks) == 1 )
        self.assertEqual( narrowpeaks[0], 20 )

        # Should only find the first peak
        narrowpeaks = find_narrow_peaks(s, threshold=1.9, peakwidth=2)
        print narrowpeaks

        self.assertTrue( len(narrowpeaks) == 1 )
        self.assertEqual( narrowpeaks[0], 10 )



def set_standing_ref():
    xdb = NVGData('/home/kjartan/Dropbox/Public/nvg201209.hdf5', mode='r+')
    xdb._pick_standing_reference("S12", "N")
    #xdb.set_standing_reference()
    xdb.hdfFile.close()



if __name__ == '__main__':
    #unittest.main()
    set_standing_ref()
