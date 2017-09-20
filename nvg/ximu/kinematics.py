""" General functions for calculating kinematics """

__version__ = '0.1'
__author__ = 'Kjartan Halvorsen'

import numpy as np
import math
import itertools
import unittest
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
import scipy.optimize as optimize
from scipy.integrate import cumtrapz
from scipy.signal import detrend, bessel, filtfilt

import scipy.io as sio
from datetime import datetime, timedelta, date

from nvg.maths import quaternions as quat
from nvg.algorithms import orientation
#from nvg.utilities import time_series
from nvg.ximu import pointfinder

from cyclicpython import cyclic_path
from cyclicpython.algorithms import kinematics as cpkinematics
#from cyclicpython.algorithms import fomatlab as fomatlab
from cyclicpython.algorithms import ekf as cpekf
from cyclicpython.algorithms import detect_peaks
from cyclicpython import cyclic_planar as cppl





def fix_cycles(ics, k=2, plotResults=False):
    """ Checks the PNAtICLA attribute, computes the
    0.25, 0.5 and 0.75 quantiles. Determines then the start and end of each cycle
    so that only cycles with length that is within median +/- k*interquartiledistance
    are kept.
    The start and end events are returned as a list of two-tuples.
    """

    if len(ics) == 0:
        warnings.warn("Unexpected empty set of events!")
        return []

    steplengths = np.array([ics[i]-ics[i-1] for i in range(1,len(ics))])
    medq = np.median(steplengths)
    q1 = np.percentile(steplengths, 25)
    q3 = np.percentile(steplengths, 75)
    interq = q3-q1
    lowthr = medq - k*interq
    #lowthr = 0.0
    highthr = medq + k*interq
    cycles = [(start_, stop_) for (stepl_, start_, stop_) \
                  in itertools.izip(steplengths, ics[:-2], ics[1:]) \
                  if (stepl_ > lowthr and stepl_ < highthr)]

    if plotResults:
        pyplot.figure()
        pyplot.hist(steplengths, 60)
        pyplot.plot([lowthr, lowthr], [0, 10], 'r')
        pyplot.plot([highthr, highthr], [0, 10], 'r')

        pyplot.hist([(stop_-start_) for (start_, stop_) in cycles], 60, color='g')

    return cycles

def angle_to_vertical(upper, lower, vertDir=[0., 0., 1], sagittalDir=None):
    """
    Calculates the angle to the vertical based on two markers upper and lower.
    If sagittalDir is provided it must be a unit vector in the direction normal
    to the sagittal plane, and the angle is the signed angle in the plane
    and taken to be positive for a positive rotation about sagittalDir to get from
    the vertical to the direction from lower to upper.
    If sagittalDir is None, the space angle (always positive) is returned.

    Arguments:
    upper, lower  -> markers (N, 3) numpy arrays
    """

    vec = upper - lower
    norms = np.apply_along_axis(np.linalg.norm, 1, vec )
    normVecT = vec.T/norms # Will be (3,N)

    if sagittalDir is None:
        costheta = np.dot(vertDir, normVecT)
        return np.arccos(costheta)

    # Find the sagittal plane.
    vecVel = np.diff(vec, axis=0)
    # These velocities lies in the sagittal plane
    (U,S,V) = np.linalg.svd(np.dot(vecVel.T, vecVel))
    sDir = V[-1]
    if np.dot(sDir, sagittalDir) < 0:
        # Flip around sDir
        sDir = -sDir


    # Calculate the angle to the vertical. A positive angle means the vertical
    # vector is rotated to the direction of the segment vec by a positive
    # rotation about sDir. This means that sDir, vertDir, vec forms a
    # right-handed triple.

    # The vertical unit vector in the sagittal plane
    vertSagittal = vert - np.dot(vert, sDir)*sDir
    vertSagittal = vertSagittal / np.linalg.norm(vertSagittal)
    # The forward unit vector in the sagittal plane
    fwd = np.cross(sDir, vertSagittal)

    # Calculating the angle
    return np.arctan2( np.dot(fwd, normVecT), np.dot(vertSagittal, normVecT) )
    
