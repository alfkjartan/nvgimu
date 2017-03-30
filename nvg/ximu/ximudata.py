""" Functions for processing data from the ximu IMU sensors. """

__version__ = '0.2'
__author__ = 'Kjartan Halvorsen'

import sys
import numpy as np
import math
import csv
import itertools
import sqlite3
import h5py
import unittest
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
import scipy.io as sio
from datetime import datetime, timedelta, date

from nvg.maths import quaternions as quat
from nvg.algorithms import orientation
#from nvg.utilities import time_series
from nvg.ximu import pointfinder

from cyclicpython import cyclic_path

def nvg_2012_09_data(dtaroot = "/media/ubuntu-15-10/home/kjartan/nvg/"):
    """
        Lists all data files for the nvg project acquired in September 2012.
        OBS: The eventlogs are missing.
    """
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


    dp = dtaroot + "2012-09-19-S5/S5"
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
    s7["LA"] = [dp + "LA-200/NVG_2012_S7_A_LA_00206_DateTime.csv", \
                    dp + "LA-200/NVG_2012_S7_A_LA_00206_CalInertialAndMag.csv"]
    s7["LH"] = [dp + "LH-800/NVG_2012_S7_A_LH_00806_DateTime.csv", \
                    dp + "LH-800/NVG_2012_S7_A_LH_00806_CalInertialAndMag.csv"]
    s7["LT"] = [dp + "LT-400/NVG_2012_S7_A_LT_00406_DateTime.csv", \
                    dp + "LT-400/NVG_2012_S7_A_LT_00406_CalInertialAndMag.csv"]
    s7["N"] = [dp + "N-600/NVG_2012_S7_A_N_00606_DateTime.csv", \
                    dp + "N-600/NVG_2012_S7_A_N_00606_CalInertialAndMag.csv"]
    s7["B"] = [dp + "B-500/NVG_2012_S7_A_B_00506_DateTime.csv", \
                    dp + "B-500/NVG_2012_S7_A_B_00506_CalInertialAndMag.csv"]
    s7["RA"] = [dp + "RA-100/NVG_2012_S7_A_RA_00106_DateTime.csv", \
                    dp + "RA-100/NVG_2012_S7_A_RA_00106_CalInertialAndMag.csv"]
    s7["RH"] = [dp + "RH-700/NVG_2012_S7_A_RH_00706_DateTime.csv", \
                    dp + "RH-700/NVG_2012_S7_A_RH_00706_CalInertialAndMag.csv"]
    s7["RT"] = [dp + "RT-300/NVG_2012_S7_A_RT_00306_DateTime.csv", \
                    dp + "RT-300/NVG_2012_S7_A_RT_00306_CalInertialAndMag.csv"]

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


    dp = dtaroot + "2012-09-24/S10/"
    s10 = {}
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


    dp = dtaroot + "2012-09-24/S11/"
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


    dp = dtaroot + "2012-09-24/S12/"
    s12 = {}
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




class NVGData:

    def __init__(self, hdfFilename):
        """ Creates the database if not already existing """
        self.fname = hdfFilename
        # No need: self.create_nvg_db(). Better to just add subject data
        self.hdfFile = h5py.File(self.fname)

        #self.rotationEstimator = CyclicEstimator(14); # Default is cyclic estimator


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
                    print("Sub group " + sstr + " already exists")
                    g = f["/"+sstr]
                    for c in ['N', 'D', 'B', 'M']:
                        try:
                            g.create_group(c)
                        except ValueError:
                            print("Sub group " + c + " already exists")

            self.hdfFile = f
        except ValueError:
            print("Error opening file " + fname)

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

    def descriptive_statistics_decorator(self, func):
        """ Decorator that will call the provided function to get data,
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

    def make_boxplot(self, results, title, ylim=None):
        """ Make a boxplot, and saves the figure as a pdf file
        using the title and date.
        The results argument is a dict with keys (subj, trial), as returned
        from a call to apply_to_all_trials.
        A tab-spaced text file with the data is also generated
        """

        # Unique list of subjects
        subjects = list(set([subj for (subj, trial) in results.keys()]))

        # Sort them
        subjects.sort(key=lambda id: int(id[1:]))

        # Order of conditions
        conds = ["N", "B", "M", "D"]

        fig= pyplot.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        ns = len(subjects)
        k=-1
        subjdta =  []
        pos = []
        midp = []
        maxp = -1e10
        subjmeans = []
        subjstds = []
        for subj in subjects:
            k += 1
            condmeans = []
            condstds = []
            for c in conds:
                try:
                    dta = results[(subj, c)]
                except:
                    # Assuming error occurs because data missing. Put in a few nans
                    print "Exception. Missing data"
                    dta = np.array([np.nan for i in range(10)])
                condmeans.append(np.mean(dta))
                condstds.append(np.std(dta))
                subjdta.append(dta)
                maxp = max(maxp, np.max(dta))
            subjmeans.append(condmeans)
            subjstds.append(condstds)
            subjpos = [5*k+i for i in range(1,5)]
            pos += subjpos
            midp.append(np.mean(np.array(subjpos))-1)
        bp = pyplot.boxplot(subjdta, positions=pos, sym='')
        pyplot.setp(bp['boxes'], color='black')
        pyplot.setp(bp['whiskers'], color='black')
        pyplot.setp(bp['fliers'], color='red', marker='+')
        xtickNames = pyplot.setp(ax, xticklabels=conds*ns)
        pyplot.setp(xtickNames, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel('Subjects and condition')
        ax.set_ylabel('Value')
        yl = ax.get_ylim()
        yl = (yl[0], maxp + 0.1*(yl[1]-yl[0]))

        if ylim == None:
            ax.set_ylim(yl)
        else:
            ax.set_ylim(ylim)
            maxp = 0.8*ylim[1]

        # Annoate with subject id
        for (subj_, pos_) in itertools.izip(subjects, midp):
            ax.text(pos_, maxp, subj_)

        pyplot.draw()

        fnameparts = title.split()
        fnameparts.append(date.today().isoformat())
        fnameparts.append(".pdf")
        pyplot.savefig("-".join(fnameparts), format="pdf")

        fnameparts.pop()
        fnameparts.append("mean")
        fnameparts.append(".tsv")
        fil = open("-".join(fnameparts), 'w')
        for c in conds:
            fil.write(c)
            fil.write('\t')
        fil.write('\n')
        np.savetxt(fil, np.array(subjmeans), delimiter="\t", newline="\n")
        fil.close()
        fnameparts.pop()
        fnameparts.pop()
        fnameparts.append("std")
        fnameparts.append(".tsv")
        fil = open("-".join(fnameparts), 'w')
        for c in conds:
            fil.write(c)
            fil.write('\t')
        fil.write('\n')
        np.savetxt(fil, np.array(subjstds), delimiter="\t", newline="\n")
        fil.close()


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
        steplengths = np.array([ics[i]-ics[i-1] for i in range(1,len(ics))])
        #medq = np.median(steplengths)
        q1 = np.percentile(steplengths, 25)
        q3 = np.percentile(steplengths, 75)
        interq = q3-q1
        #lowthr = q1 - k*interq
        lowthr = 0.0
        highthr = q3 + k*interq
        cycles = [(start_, stop_) for (stepl_, start_, stop_) \
                      in itertools.izip(steplengths, ics[:-2], ics[1:]) \
                      if (stepl_ > lowthr and stepl_ < highthr)]

        tr.attrs["PNAtCycleEvents"] = cycles
        print "%d of %d cycles discarded (%2.1f %%)" \
            % ((len(ics)-len(cycles)), len(ics),
               float(len(ics)-len(cycles))/len(ics)*100)
        pyplot.hist(steplengths, 60)
        pyplot.plot([lowthr, lowthr], [0, 10], 'r')
        pyplot.plot([highthr, highthr], [0, 10], 'r')

        pyplot.hist([(stop_-start_) for (start_, stop_) in cycles], 60, color='g')

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

    def plot_imu_data(self, subject="S7", trial="D", imu="N"):
        f = self.hdfFile
        g = f[subject]
        sg = g[trial]
        ds = sg[imu]

        pyplot.figure(figsize=(12,10))

        pyplot.subplot(3,1,1)
        pyplot.plot(ds[:,0], ds[:,1:4])
        pyplot.ylabel("Gyro [deg/s]")

        pyplot.subplot(3,1,2)
        pyplot.plot(ds[:,0], ds[:,4:7])
        pyplot.ylabel("Acc [g]")

        pyplot.subplot(3,1,3)
        pyplot.plot(ds[:,0], ds[:,7:10])
        pyplot.ylabel("Magn [G]")

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
            print "No step cycles found yet. Run detect_steps first!"
            return

        #dt = 1.0/256
        dt = 1.0/262.0 # Weird, but this is the actual sampling period

        #periods = [(cPN-pPN)*dt for (cPN, pPN) in itertools.izip(cycledta[1:], cycledta[:-1])]
        periods = [(stop_-start_)*dt for (start_, stop_) in cycledta]
        freqs = [1.0/p for p in periods]

        subj = self.hdfFile[subject]
        tr = subj[trial]
        tr.attrs['cycleFrequency'] = freqs

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
                                   startTime=5*60, anTime=60, doPlots=True):
        """ Gets the angle to the vertical for the two imus listed, then compute
        the difference in these angles.
        """
        a2v0 = self.get_angle_to_vertical(subject, trial, imus[0],
                                          startTime, anTime, False)
        a2v1 = self.get_angle_to_vertical(subject, trial, imus[1],
                                          startTime, anTime, False)

        angleBetweenSegments = []

        x = np.linspace(0,100, 100)
        for (a0, a1) in itertools.izip(a2v0, a2v1):
            # Normalize to 100 data points, then compute the difference
            a0f = a0.flatten()
            x0 = np.linspace(0,100, len(a0f))
            f0 = interp1d(x0, a0f, kind='linear')
            a0i = f0(x)

            a1f = a1.flatten()
            x1 = np.linspace(0,100, len(a1f))
            f1 = interp1d(x1, a1f, kind='linear')
            a1i = f1(x)

            angleBetweenSegments.append(a0i-a1i)

        if doPlots:
            pyplot.figure()
            for a in angleBetweenSegments:
                pyplot.plot(a*180/np.pi)
            pyplot.title("Angle between imus %s and %s for subj %s, trial %s"\
                             % (imus[0], imus[1], subject, trial))
            pyplot.ylabel('Degrees')

        subj = self.hdfFile[subject]
        tr = subj[trial]
        try:
            abi = tr.attrs['angleBetweenSegments']
        except KeyError:
            abi = []

        abi.append( ( (imus[0], imus[1]), angleBetweenSegments) )

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
                              startTime=5*60, anTime=4*60,
                              doPlots=True, sagittalPlane=True, useCyclic=14):
        """ Tracks the orientation, finds the direction of the vertical, and
        calculates the angle to the vertical. If sagittalPlane is true, then the angle
        is calculated in the x-y plane of the imu.
        Will by default use the cyclic motion method with 14 harmonics unless useCyclic=0 or
        useCyclic=False"""


        if useCyclic:
            [qimu, cycledta, imudta, cycleinds] = self.track_cyclic_orientation(useCyclic,subject, trial, imu,
                                                          startTime, anTime, False)
            print "Using cyclic orientation method"
        else:
            [qimu, cycledta, imudta, cycleinds] = self.track_orientation(subject, trial, imu,
                                                          startTime, anTime, False)

        [imuDisp, imuVel, imuGvec, cycledta, cycleinds] = self.track_displacement(subject,
                                                                                  trial,
                                                                                  imu,
                                                                                  startTime,
                                                                                  anTime,
                                                                                  False)

        a2v = []
        for (q_, g_) in itertools.izip(qimu, imuGvec):
            # g_ is in static frame coinciding with imu-frame at start of
            # each cycle.
            # The longitudinal direction of the segment is defined by the
            # x-axis (pointing downward)
            qa = quat.QuaternionArray(q_[:,1:5])
            g_.shape = (3,1)
            x_ = np.array([[-1., 0, 0]]).T
            imux_ = qa.rotateFrame(x_)

            if sagittalPlane:
                imux_[2,:] = 0.0
                g_[2,] = 0.0

            # Find the signed angle between two vectors
            gnrm = np.sqrt(np.sum(g_**2))
            gnormed = g_ / gnrm
            xdotg = np.dot(imux_.T, gnormed)
            xdotg = xdotg.flatten() / np.sqrt(np.sum(imux_**2, 0))

            # Make sure acos will work
            xdotg[np.nonzero(xdotg>1)] = 1.
            xdotg[np.nonzero(xdotg<-1)] = -1.

            gcross = np.cross(imux_.T, gnormed.T)
            sgn = np.sign(gcross[:,2])

            a2v.append(np.arccos(xdotg) * sgn.T)

        if doPlots:
            pyplot.figure()
            for a in a2v:
                pyplot.plot(a*180/np.pi)
            pyplot.title("Angle to vertical for subj %s, trial %s, imu %s"\
                             % (subject, trial, imu))

        subj = self.hdfFile[subject]
        tr = subj[trial]
        try:
            angle2vertical = tr.attrs['angle2vertical']
        except KeyError:
            angle2vertical = []

        angle2vertical.append((imu, a2v))

        return a2v



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
        """ Will first track the displacement of the imu, and then
        calculate the displacement in the vertical direction for each cycle.
        The vertical direction is determined from the average acceleration over
        each cycle
        """

        [imuDisp, imuVel, imuGvec, cycledta, cycleinds] = \
            self.track_displacement(subject, trial, imu,
                                    startTime, anTime,
                                    doPlots)

        vDisps = []
        for (d_, g_) in itertools.izip(imuDisp, imuGvec):
            vd = np.dot(d_[:,1:], g_)
            vDisps.append(np.max(vd) - np.min(vd))

        if doPlots:
            pyplot.figure()
            pyplot.hist(vDisps, bins=20)
            pyplot.title("Vertical displacment for subj %s, trial %s, imu %s"\
                             % (subject, trial, imu))

        subj = self.hdfFile[subject]
        tr = subj[trial]
        tr.attrs['verticalDisplacement'] = vDisps
        return vDisps



    def track_displacement(self, subject="S7", trial="D", imu="LA",\
                              startTime=5*60,\
                              anTime=60, doPlots=True):
        """ Will first track the orientation and the displacement of the imu,
        restarting the tracking at the beginning of each step. The direction of
        gravity is identified and the displacement is corrected for the apparent
        gravitational acceleration. The resulting displacement is returned.
        Note that a body-fixed (imu-fixed) coordinate system is used.
        """

        [qimu, cycledta, imudta, cycleinds] = self.track_orientation(subject, trial, imu,
                                                          startTime, anTime, False)

        aimu = []
        for (imudta_, qimu_) in itertools.izip(imudta, qimu):
            accimu = imudta_[:, [0,4,5,6]].copy()
            accimu[:, 1:] *= 9.82

            # Rotate acceleration vectors
            qaimu = quat.QuaternionArray(qimu_[:,1:5])
            accimu[:,1:] = (qaimu.rotateFrame(accimu[:,1:].T)).T
            aimu.append(accimu)
        # Integrate by trapezoidal rule
        [dimu, vimu, gvecs] = _integrate_acc(aimu, cycledta)

        if doPlots: # Check results
            pyplot.figure()
            pyplot.subplot(3,1,1)
            for d_ in dimu:
                pyplot.plot(d_[:,0], d_[:,1], 'b')
                pyplot.plot(d_[:,0], d_[:,2], 'g')
                pyplot.plot(d_[:,0], d_[:,3], 'r')
            yl = (-0.3, 0.3)
            for (s_, e_) in cycledta:
                pyplot.plot([s_, s_], yl, 'm')
                pyplot.plot([e_, e_], yl, 'c')
            pyplot.ylim(yl)

            pyplot.subplot(3,1,2)
            for v_ in vimu:
                pyplot.plot(v_[:,0], v_[:,1], 'b')
                pyplot.plot(v_[:,0], v_[:,2], 'g')
                pyplot.plot(v_[:,0], v_[:,3], 'r')
            yl = (-2, 2)
            for (s_, e_) in cycledta:
                pyplot.plot([s_, s_], yl, 'm')
                pyplot.plot([e_, e_], yl, 'c')
            pyplot.ylim(yl)

            pyplot.subplot(3,1,3)
            for a_ in aimu:
                pyplot.plot(a_[:,0], a_[:,1], 'b')
                pyplot.plot(a_[:,0], a_[:,2], 'g')
                pyplot.plot(a_[:,0], a_[:,3], 'r')
            yl = (-30, 30)
            for (s_, e_) in cycledta:
                pyplot.plot([s_, s_], yl, 'm')
                pyplot.plot([e_, e_], yl, 'c')
            pyplot.ylim(yl)

        return [dimu, vimu, gvecs, cycledta, cycleinds]


    def track_orientation(self, subject="S7", trial="D", imu="LA",\
                              startTime=5*60,\
                              anTime=60, doPlots=True):
        """ Will track the orientation of the imu. Assumes that the start of each cycle
        is detected and available in the trial attribute 'PNAtICLA'.
        The tracking algorithm is restarted at each step. So, for each timestep,
        a quaternion is estimated that describes the orientation of the IMU w.r.t. the
        orientation at the initical contact of the current gait cycle.
        """

        [imudta, tr, subj] = self.get_imu_data(subject, trial, imu, startTime, anTime)
        firstPN = imudta[0,0]
        lastPN = imudta[-1,0]

        try:
            cycledta = self.get_cycle_data(subject, trial, imu, firstPN, lastPN)
        except KeyError:
            print "No step cycles found yet. Run detect_steps first!"

        if doPlots: # Check results
            pyplot.figure()
            pyplot.plot(imudta[:,0], imudta[:,4:7])
            for ind in cycledta:
                pyplot.plot([ind, ind], [-5, 5], 'm')

        # Use tracking algorithm from nvg. Restart at each cycle start
        imuq = []
        cycleinds = []
        imuDataSplit = []
        for (cstart, cstop) in cycledta:
            (indStart,) = np.nonzero(imudta[:,0] < cstart)
            (indEnd,) = np.nonzero(imudta[:,0] > cstop)
            indx = np.ix_(range(indStart[-1], indEnd[1]), range(imudta.shape[1]))
            imud = imudta[indx]
            #[q, cinds] = self.get_orientation(imud, [cstart, cstop], doPlots)
            [q, cinds] = self.orientationEstimator.estimatea(imud, [cstart, cstop], doPlots)
            imuq.append(q)
            cycleinds.append(cinds)
            imuDataSplit.append(imud)

            #[imuq, cycleinds] = self.get_orientation(imudta, cycledta, doPlots)
        #return [imuq, cycledta, imudta, cycleinds]
        return [imuq, cycledta, imuDataSplit, cycleinds]

    def track_cyclic_orientation(self, nHarmonics=14, subject="S7", trial="D", imu="LA",\
                              startTime=5*60,\
                              anTime=60, doPlots=True):
        """ Will track the orientation of the imu using a model of the orientation as a
        truncated Fourier series. Assumes that the start of each cycle
        is detected and available in the trial attribute 'PNAtICLA'.
        The tracking algorithm is restarted at each step. So, for each timestep,
        a quaternion is estimated that describes the orientation of the IMU w.r.t. the
        orientation at the initical contact of the current gait cycle.
        """

        [imudta, tr, subj] = self.get_imu_data(subject, trial, imu, startTime, anTime)
        firstPN = imudta[0,0]
        lastPN = imudta[-1,0]

        try:
            cycledta = self.get_cycle_data(subject, trial, imu, firstPN, lastPN)
        except KeyError:
            print "No step cycles found yet. Run detect_steps first!"

        if doPlots: # Check results
            pyplot.figure()
            pyplot.plot(imudta[:,0], imudta[:,4:7])
            for ind in cycledta:
                pyplot.plot([ind, ind], [-5, 5], 'm')

        # Use cyclic tracking algorithm
        imuq = []
        cycleinds = []
        imuDataSplit = []
        for (cstart, cstop) in cycledta:
            (indStart,) = np.nonzero(imudta[:,0] < cstart)
            (indEnd,) = np.nonzero(imudta[:,0] > cstop)
            indx = np.ix_(range(indStart[-1], indEnd[1]), range(imudta.shape[1]))
            imud = imudta[indx]
            q = self.get_cyclic_orientation(imud, nHarmonics)
            imuq.append(q)
            #cycleinds.append(cinds)
            imuDataSplit.append(imud)

        return [imuq, cycledta, imud, cycleinds]


    def get_cycle_data(self, subject, trial, imu, firstPN, lastPN):
        syncimu = self.get_PN_at_sync(subject,imu)
        syncLA = self.get_PN_at_sync(subject,"LA")
        subj = self.hdfFile[subject]
        tr = subj[trial]

        #cycledtaold = [ind-syncLA[0]+syncimu[0] for ind in tr.attrs["PNAtICLA"] if \
        #                ind-syncLA[0] > firstPN-syncimu[0] \
        #                and ind-syncLA[0] < lastPN-syncimu[0]]

        cycledta = [(start_-syncLA[0]+syncimu[0], stop_-syncLA[0]+syncimu[0]) \
                        for (start_, stop_) in tr.attrs["PNAtCycleEvents"] \
                        if start_-syncLA[0] > firstPN-syncimu[0] \
                        and stop_-syncLA[0] < lastPN-syncimu[0]]

        return cycledta


    def get_cyclic_orientation(self, imudta, nHarmonics):
        """
        Runs the cyclic orientation method assuming that the imud is a single cycledta
        """
        dt = 1.0/256.0
        tvec = imudta[:,0]*dt

        #accdta = imudta[:,4:7]*9.82
        gyrodta = imudta[:,1:4]*np.pi/180.0
        magdta = imudta[:,7:10]

        omega = 2*np.pi/ (tvec[-1]  - tvec[0])

        (qEst, bEst) = cyclic_path.estimate_cyclic_orientation(tvec, gyrodta, magdta, omega, nHarmonics)
        tvec.shape = (len(tvec), 1)
        return np.hstack((tvec, qEst))


    def get_orientation(self, imudta, cycledta, doPlots):
        imuq = np.zeros((imudta.shape[0], 5))
        initRotation = quat.Quaternion(1,0,0,0)
        initRotVelocity = np.zeros((3,1))
        initCov = np.diag(np.array([1, 1, 1, 1e-1, 1e-1, 1e-1, 1e-1]))
        measuremCov = np.diag(np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]))
        procNoiseVar = 0.5 # rad^2/s^2
        tau = 0.5 # s, time constant of cont time model of movement
        orientFilter = None
        dt = 1/256.0 # s per packet number
        gMagn = 9.82
        deg2rad = np.pi/180.0
        imudtaT = imudta.T
        cycleind = 0
        cycledta.append(1e+12)
        cycledtainds = []
        for i in range(imuq.shape[0]):
            t = imudta[i,0]*dt
            if (int(imudta[i,0]) >= cycledta[cycleind]) or i == 0:
                # Start of new cycle
                #if orientFilter != None:
                    #initialRotVelocity = orientFilter.rotationalVelocity.latestValue
                    #initCov = np.diag(np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]))

                orientFilter = orientation.GyroIntegrator(t, initRotation)
                #initRotVelocity,
                #                         initCov, measuremCov, procNoiseVar, tau)
                if i != 0:
                    cycleind += 1
                    cycledtainds.append(i)
            else:

                orientFilter(imudtaT[4:7,i]*gMagn, imudtaT[7:10,i], imudtaT[1:4,i]*deg2rad, t)

            imuq[i,0] = t
            imuq[i,1] = orientFilter.rotation.latestValue.w
            imuq[i,2] = orientFilter.rotation.latestValue.x
            imuq[i,3] = orientFilter.rotation.latestValue.y
            imuq[i,4] = orientFilter.rotation.latestValue.z

        cycledta.pop()
        if doPlots: # Check results
            pyplot.figure()
            pyplot.plot(imuq[:,0], imuq[:,1:5])
            for ind in cycledta:
                pyplot.plot([dt*ind, dt*ind], [-1, 1], 'm')

        return [imuq, cycledtainds]

    def get_imu_data(self, subject="S7", trial="D", imu="LH",\
                         startTime=0,\
                         anTime=600, rawData=nvg_2012_09_data):
        """ Returns data for the specified subject, trial and imu.
        The data returned starts at the specified time into the trial, and has the
        specified length in seconds.

        New 2017-03-30: If startTime is negative, look up raw data file and
        read from this.
        """

        subj = self.hdfFile[subject]
        tr = subj[trial]
        #PNperSecond = self.hdfFile.attrs['PNperSecond']
        sfreq = self.hdfFile.attrs['packetNumbersPerSecond']

        imudta = tr[imu]

        if startTime >= 0:
            # There could be a few packets missing, but we ignore this at the moment.
            return (imudta[startTime*sfreq:(startTime+anTime)*sfreq,:], tr, subj)
        else:
            (dta, events) = rawData()
            fnames = dta[subject][imu] # (DateTime.csv, CalInertialAndMag.csv)
            startPacket = startTime*sfreq + imudta[0,0]
            endPacket = anTime*sfreq + imudta[0,0]
            return (read_packets(fnames[1], startPacket, endPacket), tr, subj)

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
        subj = self.hdfFile[subject]
        PNs = subj.attrs['PNAtSync']
        if PNs.shape[0] == 7:
            # No B data
            keys = ["RT", "LA", "LH", "N", "LT", "RA", "RH"]
        else:
            keys = ["RT", "B", "LA", "LH", "N", "LT", "RA", "RH"]

        ind = keys.index(imu)
        return PNs[ind]

def _integrate_acc(acc, cycledta):
    """ Integrate acceleration to get velocity and displacement. The
    integration is done using the trapezoidal rule, and reset at
    the start of each cycle. The mean acceleration and mean displacement
    are removed during each cycle.
    """

    #[imuV, gvecs] = _integrate_cyclic(acc, cycledta)
    #[imuD, slask] = _integrate_cyclic(imuV, cycledta)
    #return [imuD, imuV, gvecs]

    gvecs = []
    imuD = []
    imuV = []
    for (acc_, c_) in itertools.izip(acc, cycledta):
        cl = list(c_)
        [v, g] = _integrate_cyclic(acc_, cl)
        [d, slask] = _integrate_cyclic(v, cl)
        gvecs.append(g[-1])
        imuD.append(d)
        imuV.append(v)

    return [imuD, imuV, gvecs]


def _integrate_cyclic(a, cycledta):

    nfrs = a.shape[0]
    v = np.zeros((nfrs, 4))
    v[:,0] = a[:,0]
    currentV = np.array([0.0, 0.0, 0.0])
    meanA = np.array([0.0, 0.0, 0.0])
    tau = 0.0
    cycleIndAndT = []
    h = 1/256.0
    lastT = a[0,0]*h
    cycledta.append(1e12)
    cycleind = 0
    gvecs = []
    for i in range(1,nfrs):
        ind = a[i,0]
        if ind >= cycledta[cycleind]:
            if (tau>0):
                meanA /= tau
                gvecs.append(meanA/np.linalg.norm(meanA))
            for (ii, tt) in cycleIndAndT:
                v[ii,1:4] -= meanA*tt
            currentV[:] = 0.0
            meanA[:] = 0.0
            cycleIndAndT = []
            tau = 0.0
            cycleind += 1
        t = ind*h
        dt = t-lastT
        lastT = t
        cycleIndAndT.append((i, tau))
        tau += dt
        meanA += dt*a[i,1:4]
        currentV += dt*0.5*(a[i-1, 1:4] + a[i, 1:4])
        v[i,1:4] = currentV

    cycledta.pop()
    return [v, gvecs]

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
                print "DEBUG. Found %d peaks using threshold %1.3f" % (len(indxs), thr)
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
                print "Deviating time of sync in signal %d: %d seconds"% (ol, tsdev[ol])
                pn = packetNumbers[ol]
                elindx = range(pn[1]-10, pn[1]+10)
                s = mysignals[ol]
                s[np.ix_(elindx, [1])] = 0
                if len(outliers[0]) == 0:
                    concensus = True
        else:
            concensus = True
    if not concensus:
        print "Could not find timestamps to agree at sync pulse \
            Could be due to wrong setting of time on imu."

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
            print "Unexpected large acceleration (%1.4f) at beginning for imu %s " \
                % (accmagn[0:10].mean(), imu)

        if accmagn[-10:].mean() > 0.2:
            print "Unexpected large acceleration (%1.4f) at end for imu %s " \
                % (accmagn[-10:].mean(), imu)


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



if __name__ == '__main__':
    unittest.main()
