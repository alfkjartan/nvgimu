""" Functions for processing marker data. """

__version__ = '0.1'
__author__ = 'Kjartan Halvorsen'

import numpy as np
import math
import csv
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
from nvg.ximu import kinematics
from nvg.io import qualisys_tsv as qtsv

from cyclicpython import cyclic_path
from cyclicpython.algorithms import kinematics as cpkinematics
#from cyclicpython.algorithms import fomatlab as fomatlab
from cyclicpython.algorithms import ekf as cpekf
from cyclicpython.algorithms import detect_peaks
from cyclicpython import cyclic_planar as cppl



def get_marker_data_cycles(md, frames2use):
    """
    Will find start and end of gait cycles using the trajectory of the left
    ankle marker

    Arguments
    md         ->  marker data object
    frames2use ->  list of timestamps

    Returns
    list of tuples (startTime, endTime) of each cycle
    list of tuples (startInd, endInd) of each cycle
    """


    # Find initical contact from ankle marker data

    ankled = md.marker('ANKLE').position(frames2use)
    ics = detect_heel_strike(frames2use, ankled[2,:], plotResults=plotResults)
    cycledtaInds = kinematics.fix_cycles(ics, k=0.5,  plotResults=plotResults)
    cycledtaTimes = [(frames2use[start_], frames2use[end_])
                            for (start_, end_) in cycledtaInds]

    return (cycledtaTimes, cycledtaInds)



def detect_heel_strike(tvec, ankleposz, wn=0.2, posThr=[0.03, 0.08],
                        velThr = [-100, 0], accThr = [5, 100],
                         plotResults=False):
    """
    Returns a list of heelstrikes detected from the z-component of the ankle
    marker. From the ankle position, velocity and acceleration are computed.
    The following heuristic is used to detect the heel strike:
    minpos + posThr[0] < position < minpos + posThr[1]
    velThr[0] < velocity < velThr[1]
    accThr[0] < acc < accThr[1]

    Arguments:
    tvec        ->  Time vector (N,)
    ankleposz   ->  ankle position in vertical direction (N,)
    wn          ->  cutoff frequency of the low pass filter (Nyquist freq = 1)
    plotResults ->  If true do plot

    Returns:
    pks         <-  list    of indices where peaks are found
    """

    # Lowpass filter using a Bessel filter
    [b,a] = bessel(4, wn)
    ap = filtfilt(b,a,ankleposz)

    dt = np.mean(np.diff(tvec))
    av = np.diff(ap)/dt
    aa = np.diff(av)/dt

    apmin = np.min(ap)

    okinds = np.where( np.logical_and( np.logical_and(
                        np.logical_and( ap[:-2] > (apmin + posThr[0]),
                                       ap[:-2] < (apmin + posThr[1])),
                        np.logical_and( av[:-1] > velThr[0],
                                       av[:-1] < velThr[1])),
                        np.logical_and( aa > accThr[0],
                                       aa < accThr[1])))




    aaa = np.empty(aa.shape)
    aaa[:] = np.nan
    aaa[okinds] = aa[okinds]

    #pks = detect_peaks.detect_peaks(aa, mph=10, mpd=10)
    pks = detect_peaks.detect_peaks(aaa, mph=5, mpd=40)
    pks = np.intersect1d(pks, okinds)

    if plotResults:
        pyplot.figure()
        pyplot.subplot(3,1,1)
        pyplot.plot(tvec, ankleposz, alpha=0.3)
        pyplot.plot(tvec, ap)
        for ic_ in pks:
            pyplot.plot([tvec[ic_], tvec[ic_]], [-0.3, 0], 'm', alpha=0.5)
        pyplot.plot([tvec[0], tvec[-1]], [apmin+posThr[0], apmin+posThr[0]], 'y')
        pyplot.plot([tvec[0], tvec[-1]], [apmin+posThr[1], apmin+posThr[1]], 'c')
        pyplot.ylim((-0.3, -0.1))

        pyplot.subplot(3,1,2)
        pyplot.plot(tvec[:-1], av)
        for ic_ in pks:
            pyplot.plot([tvec[ic_], tvec[ic_]], [-1, 1], 'm', alpha=0.6)
        pyplot.plot([tvec[0], tvec[-1]], [velThr[0], velThr[0]], 'y')
        pyplot.plot([tvec[0], tvec[-1]], [velThr[1], velThr[1]], 'c')
        pyplot.ylim((-1, 1))

        pyplot.subplot(3,1,3)
        pyplot.plot(tvec[:-2], aa)
        for ic_ in pks:
            pyplot.plot([tvec[ic_], tvec[ic_]], [-10, 10], 'm', alpha=0.6)
        pyplot.plot([tvec[0], tvec[-1]], [accThr[0], accThr[0]], 'y')
        pyplot.plot([tvec[0], tvec[-1]], [accThr[1], accThr[1]], 'c')
        pyplot.ylim((-10, 10))

    return pks



def split_in_cycles(tvec, dta, cycledta, indices=False, minCycleLength=80):
    """ Will split the data matrix dta (timeseries in columns) into cycles given
    in cycledta.

    Arguments:
    tvec     -> time vector (N,)
    dta      -> data matrix (N,m)
    cycledta -> list of (start,stop) times corresponding to the times in tvec.
                OR, indices (start, stop)
    indices  -> if True, then cycledta contains indices, not times.

    Returns tuple:
    timespl  <- the time vector split in cycles, list
    dtaspl   <- the data matrix split in cycles, list
    """
    timespl = []
    dtaspl = []
    tv = np.asarray(tvec).ravel() # Make sure it is a numpy 1d-array
    for (cstart,cstop) in cycledta:
        if indices:
            indStart = [cstart]
            indEnd = [cstop]
        else:
            (indStart,) = np.nonzero(tv < cstart)
            (indEnd,) = np.nonzero(tv > cstop)
            if len(indStart) == 0:
                indStart = [0]
            if len(indEnd) == 0:
                indEnd = [len(tv)-1]

        if indEnd[0] - indStart[-1]  > minCycleLength:
            timespl.append(tv[indStart[-1]:indEnd[0]])

            if dta.ndim == 1:
                dtaspl.append(dta[indStart[-1]:indEnd[0]])
            else:
                dtaspl.append(dta[indStart[-1]:indEnd[0],:])

    return (timespl, dtaspl)

def resample_timeseries(x, t, tnew, kind='linear'):
    """ Resamples the timeseries x defined at times t by interpolation """
    f = interp1d(t, x, kind=kind, axis=0)
    return f(tnew)


def check_sync(syncfilename="/home/kjartan/Dropbox/projekt/nvg/data/solna09/S7/NVG_2012_S7_sync.tsv"):
    """
        Loads the tsv file with markerdata from the synchronization experiment. Plots the z-coordinate of the marker
        'clapper1'.
    """
    title = "Checking sync of file %s" %syncfilename

    md = qtsv.loadQualisysTSVFile(syncfilename)
    timeToSync = md.syncTime - md.timeStamp

    clapper = md.marker('clapper1')
    clapposz = clapper.position(md.frameTimes).transpose()[:,2]

    plt.figure()
    plt.plot(md.frameTimes, clapposz)
    plt.plot(timeToSync.total_seconds()*np.array([1, 1]), [-0.3, 1])
    plt.title(title)


def test_three_point_angle():
    p0 = np.array([0.0,0,0])
    p1 = np.array([1.0,0,0])
    p2 = np.array([1.0,1.0,0])
    p3 = np.array([0.0,2.0,0])

    npt.assert_almost_equal( _three_point_angle(p1,p0,p2,np.array([0,0,1.0])), np.pi/4)
    npt.assert_almost_equal( _three_point_angle_projected(p1,p0,p2,np.array([0,0,1.0])), np.pi/4 )
    npt.assert_almost_equal( _three_point_angle(p1,p0,p3,np.array([0,0,1.0])), np.pi/2)
    npt.assert_almost_equal( _three_point_angle_projected(p1,p0,p3,np.array([0,0,1.0])), np.pi/2 )

def _four_point_angle(pp1, pp2, pd1, pd2, posdir):
    """
    Computes the angle between the lines pp1-pp2 and pd1-pd2.
      posdir is a vector in 3D giving the positive direction of rotation, using the right-hand rule. The angle is measured from 0 to 360 degrees as a rotation from (p1-pcentral) to (p2-pcentral).
    """

    v1 = pp1-pp2
    v2 = pd1-pd2

    return _two_vec_angle(v1,v2,posdir)

def _three_point_angle(p1, pcentral, p2, posdir):
    """
    Will compute the angle between the three points, using pcentral as the center.
    posdir is a vector in 3D giving the positive direction of rotation, using the right-hand rule. The angle is measured from 0 to 360 degrees as a rotation from (p1-pcentral) to (p2-pcentral).
    """
    v1 = p1-pcentral
    v2 = -(p2-pcentral)

    return _two_vec_angle(v1,v2,posdir)

def _two_vec_angle(v1,v2,posdir):

    if v1.ndim == 1:
        v1.shape += (1,)
        v2.shape += (1,)

    theta = np.zeros((v1.shape[1],))

    for i in range(v1.shape[1]):
        v1_ = v1[:,i]
        v2_ = v2[:,i]

        theta[i] = np.arccos( np.inner(v1_, v2_) / np.linalg.norm(v1_) / np.linalg.norm(v2_) )
        v3_ = np.cross(v1_,v2_)
        if (np.inner(v3_, posdir) < 0):
            #theta[i] = 2*np.pi - theta[i]
            theta[i] = - theta[i]

    return theta

def _four_point_angle_projected(pp1, pp2, pd1, pd2, posdir):
    """
    Computes the angle between the lines pp1-pp2 and pd1-pd2.
      posdir is a vector in 3D giving the positive direction of rotation, using the right-hand rule. The angle is measured from 0 to 360 degrees as a rotation from (p1-pcentral) to (p2-pcentral).
    """

    v1 = pp1-pp2
    v2 = pd1-pd2

    return _two_vec_angle_projected(v1,v2,posdir)

def _three_point_angle_projected(p1, pcentral, p2, posdir):
    """
    Will compute the angle between the three points, using pcentral as the center.
    posdir is a vector in 3D giving the positive direction of rotation, using the right-hand rule. The angle is measured from 0 to 360 degrees as a rotation from (p1-pcentral) to (p2-pcentral).
    """

    v1 = p1-pcentral
    v2 = p2-pcentral

    return _two_vec_angle_projected(v1,v2,posdir)

def _two_vec_angle_projected(v1,v2,posdir):

    Pr = np.identity(3) - np.outer(posdir,posdir)


    if v1.ndim == 1:
        v1.shape += (1,)
        v2.shape += (1,)

    theta = np.zeros((v1.shape[1],))

    for i in range(v1.shape[1]):
        v1_ = np.dot( Pr, v1[:,i] )
        v2_ = np.dot( Pr, v2[:,i] )

        theta[i] = np.arccos( np.inner(v1_, v2_) / np.linalg.norm(v1_) / np.linalg.norm(v2_) )
        v3_ = np.cross(v1_,v2_)
        if (np.inner(v3_, posdir) < 0):
            theta[i] = 2*np.pi - theta[i]

    return theta
