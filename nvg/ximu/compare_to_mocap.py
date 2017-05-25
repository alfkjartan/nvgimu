# -*- coding: utf-8 -*-
"""
Loads Qualisys motion capture data and IMU data and makes comparison
"""
# Copyright (C) 2014-- Kjartan Halvorsen
#
# This file is part of nvgimu
#
# nvgimu is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Nvg is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with nvgimu.  If not, see <http://www.gnu.org/licenses/>.

from datetime import datetime, timedelta, date
import os
import numpy.testing as npt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import itertools
from nvg.io import qualisys_tsv as qtsv
from nvg.ximu import ximudata as xdt
from cyclicpython.algorithms import detect_peaks

# Load the imu data so that it is accessible as a global variable in this module
xdb = xdt.NVGData('/home/kjartan/Dropbox/Public/nvg201209.hdf5')



def get_comparison_data(mocapdatafile, markers,  subject, trial, imus,
                        startTime=60, anTime=180):
    """ Returns imu data and mocapdata

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
        (imudt, subj_, trial_) = xdb.get_imu_data(subject, trial, imu, startTime, anTime)
        imudata[imu] =  imudt


    (imudt, subj_, trial_) = xdb.get_imu_data(subject, trial, "LA", startTime, anTime)
    firstPN = imudt[0,0]
    lastPN = imudt[-1,0]
    syncLA = xdb.get_PN_at_sync(subject, "LA")
    dt = 1.0/262.0 # Weird, but this is the actual sampling period
    tvec = dt*(imudt[:,0]-syncLA[0])
    cycledtaNoShift= xdb.get_cycle_data(subject, trial, "LA", firstPN, lastPN)

    packetNumbers = np.asarray(imudt[:,0], np.int32)
    cycledtaInds = [ (np.where(packetNumbers <= cd_[0])[0][-1],
                      np.where(packetNumbers >= cd_[1])[0][0])
                        for cd_ in cycledtaNoShift ]

    cycledtaSec = [ ( (cd_[0]-syncLA[0])*dt, (cd_[1]-syncLA[0])*dt )
                        for cd_ in cycledtaNoShift]

    imudata['cycledata'] = cycledtaNoShift # PNs of LA imu at events
    imudata['cycledataSec'] = cycledtaSec # times in seconds since sync
    imudata['cycledataInds'] = cycledtaInds # times in seconds since sync

    if isinstance(mocapdatafile, basestring):
        md = qtsv.loadQualisysTSVFile(mocapdatafile)
    else:
        md = mocapdatafile

    timeSinceSync = md.timeStamp - md.syncTime

    firstCycleStart = cycledtaSec[0][0]
    lastCycleEnd = cycledtaSec[-1][-1]

    frames2use = md.frameTimes[md.frameTimes>firstCycleStart-timeSinceSync.total_seconds()]
    frames2use = frames2use[frames2use<lastCycleEnd-timeSinceSync.total_seconds()]
    ft = frames2use + timeSinceSync.total_seconds()

    markerdata = {'frames':frames2use}
    markerdata['frametimes'] = ft
    for m in markers:
        markerdata[m] = md.marker(m).position(frames2use)


    # Find initical contact from ankle marker data
    ankled = md.marker('ANKLE').position(frames2use)
    ics = detect_heel_strike(ft, ankled[2,:], plotResults=False)
    cycledtaMC = xdt.fix_cycles(ics, k=1,  plotResults=False)
    markerdata['cycledataMC'] = cycledtaMC

    return (markerdata, imudata)

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


def compare_angle_to_vertical(mocapdata, subject, trial="N",
                    resdir="/home/kjartan/Dropbox/project/nvg/resultat/compare-mocap",
                    startTime = 60,
                    anTime = 240,
                    imu="LA", markers=["ANKLE", "KNEE"],
                    peakat=70,
                    resetAtIC=False,
                    var_angvel=1,
                    var_incl=1e-2,
                    lambda_gyro=10,
                    lambda_incl=0.1,
                    gThreshold=5e-1,
                    useCyclic=True):
    """
    Compares angle to the vertical of given IMU and vector from first to second
    of the markers given """

    (md, imudt) = get_comparison_data(mocapdata, markers,
                                subject, trial, [imu],
                                startTime, anTime)
    distd = md[markers[0]]
    proxd = md[markers[1]]


    # Find the sagittal plane.
    llVec = proxd.T - distd.T
    llVecVel = np.diff(llVec, axis=0)
    # These velocities lies in the sagittal plane
    (U,S,V) = np.linalg.svd(np.dot(llVec.T, llVec))
    sagittalDir = V[-1]

    # If not in direction of global negative x, then flip
    if sagittalDir[0] > 0:
        sagittalDir = -sagittalDir


    # Calculate the angle to the vertical. Seen from the lefft of the person,
    # a positive angle means the knee is in front of the ankle, or
    # equivalently, a positive angle is a positive rotation about a laterally
    # pointing axis of the ankle, with angle zero meaning that the lower leg
    # is vertical.

    # The vertical unit vector in the sagittal plane
    vert = np.array([0., 0., 1.])
    vertSagittal = vert - np.dot(vert, sagittalDir)*sagittalDir
    vertSagittal = vertSagittal / np.linalg.norm(vertSagittal)
    # The backward unit vector in the sagittal plane
    bwd = np.cross(vertSagittal, sagittalDir)
    print "Sagittal direction and backwards from markers"
    print vertSagittal
    print bwd


    # Calculating the angle
    ll_angle = np.arctan2( -np.dot(llVec, bwd), np.dot(llVec, vertSagittal) )

    ft = md['frametimes']
    cycledta = imudt['cycledataSec']

    #Find cycles from marker data since cannot trust sync between mocap and imu
    cycledtaMC =md['cycledataMC']
    #OBS: These are indices into the rows of the data matrix

    # Split joint angle from markers .
    (timeSplit, angleSplitNoNorm) = split_in_cycles(ft, ll_angle,
                                                    cycledtaMC, indices=True)

    # normalize to 100 datapoints
    tnew = np.linspace(0,99, 100)
    angleSplit = []
    for ja in angleSplitNoNorm:
        t0 = np.linspace(0,99, len(ja))
        angleSplit.append(resample_timeseries(ja, t0, tnew))

    # Now the angle to vertical from the IMUs
    # Estimate the sagittal plane axis in local IMU
    jointIMUPair = {"LA":"LT", "LT":"LA", "RA":"RT", "RT":"RA"}

    (d0, d1) = xdb.estimate_joint_axes(subject, trial,
                                        [imu, jointIMUPair[imu]],
                                        startTime, anTime)
    # If left hand-side, the sagittal direction should be in the positive
    # local z direction
    if imu[0] == "L":
        if d0[2] < 0:
            d0 = -d0

    # If it is on the right hand side, in the negative local z-direction
    if imu[0] == "R":
        if d0[2] > 0:
            d0 = -d0

    print "Saggital direction from IMU"
    print d0

    angleSplit_imu = []

    if useCyclic:
        if resetAtIC:
            omega=None
        else:
            cyclestartstop = np.asarray(cycledta)
            cyclePeriod = np.median(cyclestartstop[:,1] - cyclestartstop[:,0])
            omega = 2*np.pi/cyclePeriod
            print "Cycle period: %f" %cyclePeriod

        angleTracker = xdb.angle_to_vertical_cyclic_tracker(omega=omega,
                                                        nHarmonics=16,
                                                        sagittalDir=d0,
                                                        var_angvel=var_angvel,
                                                        var_incl=var_incl,
                                                        lambda_gyro=lambda_gyro,
                                                        lambda_incl=lambda_incl)
    else:
        angleTracker = xdb.angle_to_vertical_ekf_tracker(sagittalDir,
                                                        var_angvel=var_angvel,
                                                        var_incl=var_incl,
                                                        m=20)


    angle2verticalSplit = xdb.get_angle_to_vertical(subject, trial,
                                                imu,
                                                startTime=startTime,
                                                anTime=anTime,
                                                angleTracker=angleTracker,
                                                resetAtIC=resetAtIC,
                                                gThreshold=gThreshold)


    plt.figure()
    plt.plot(angleTracker.link.yinc[:,0], angleTracker.link.yinc[:,1], 'o')
    plt.plot(angleTracker.link.tvec, angleTracker.link.phi)

    plt.figure()
    plt.plot(angleTracker.link.tvec, angleTracker.link.gyrodta)

    for ja in angle2verticalSplit:
        t0 = np.linspace(0,99, len(ja))
        angleSplit_imu.append(resample_timeseries(ja, t0, tnew))



    angle_md = []
    tvec = np.linspace(0,99,100)
    #1/0
    for ja in angleSplit:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        if peakind < peakat+20 and peakind > peakat-20:
            t = tvec - peakind
        else:
            t = tvec - peakat
        # Look for valley in cycle before expected positive peak
        (negpeakind,) = np.nonzero(ja == np.min(ja[:peakat]))

        negpeakind = negpeakind[0]
        #angle = ja - ja[negpeakind]
        angle = ja
        angle_md.append( (angle, t, peakind) )
        #t = tvec

    angle_imu = []
    for ja in angleSplit_imu:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        if peakind < peakat+20 and peakind > peakat-20:
            t = tvec - peakind
        else:
            t = tvec - peakat
        (negpeakind,) = np.nonzero(ja == np.min(ja))
        negpeakind = negpeakind[0]
        #angle = ja - ja[negpeakind]
        angle = ja
        angle_imu.append( (angle, t, peakind) )
        #t = tvec


    angle_md_nooutliers = fix_outliers(angle_md)
    angle_imu_nooutliers = fix_outliers(angle_imu)

    plt.figure()
    for ja in angle_md_nooutliers:
        plt.plot(ja[1], ja[0]*180/np.pi)
    plt.title("Angle to vertical from marker data " + subject + ' trial ' + trial )
    plt.ylabel("Degrees")
    plt.savefig(resdir + '/' + subject + '_' + trial + '_' + imu
                    + '_angle_to_vertical_mocap.pdf')

    #1/0
    plt.figure()
    for ja in angle_imu_nooutliers:
        plt.plot(ja[1], ja[0]*180/np.pi)
    plt.title("Angle from imu data " + subject + ' trial ' + trial )
    plt.ylabel("Degrees")
    plt.savefig(resdir + '/' + subject + '_' + trial + '_' + imu
                + '_angle_to_vertical_imu.pdf')

    _stats_plot(angle_md_nooutliers, angle_imu_nooutliers, peakat)
    plt.title('Angle to vertical of lower leg. Ensemble mean +/- 2 std  ' + subject + ' trial ' + trial )
    plt.ylabel('Degrees')
    plt.xticks([])
    plt.savefig(resdir + '/' + subject + '_' + trial + '_' + imu
                + '_angle_to_vertical-mean-std.pdf')


def compare_knee_RoM(mocapdata, subject, trial="N",
                    resdir="/home/kjartan/Dropbox/project/nvg/resultat/compare-mocap",
                    startTime = 60,
                    anTime = 240,
                    peakat=70):
    """  Compares Range of motion of knee flexion for trial N (Normal condition). """

    (md, imudt) = get_comparison_data(mocapdata,
                                ['ANKLE', 'KNEE', 'THIGH', 'HIP'],
                                subject, trial, ['LA'],
                                startTime, anTime)
    ankled = md['ANKLE']
    kneed = md['KNEE']
    thighd = md['THIGH']
    hipd = md['HIP']

    jointangle = _three_point_angle( hipd, kneed, ankled, np.array([-1.0,0,0]) )
    #jointangle = _three_point_angle_projected( thighd, kneed, ankled, np.array([-1.0,0,0]) )
    #jointangle = _three_point_angle_projected( hipd, kneed, ankled, np.array([-1.0,0,0]) )
    #jointangle = _four_point_angle_projected( hipd, thighd, ankled, kneed, np.array([-1.0,0,0]) )

    ft = md['frametimes']
    cycledta = imudt['cycledataSec']

    #Find cycles from marker data since cannot trust sync between mocap and imu
    cycledtaMC =md['cycledataMC']
    #OBS: These are indices into the rows of the data matrix

    # Split joint angle from markers and normalize to 100 datapoints.
    (timeSplit, jointangleSplitNoNorm) = split_in_cycles(ft, jointangle,
                                                    cycledtaMC, indices=True)

    #1/0
    tnew = np.linspace(0,99, 100)
    jointangleSplit = []
    for ja in jointangleSplitNoNorm:

        t0 = np.linspace(0,99, len(ja))
        jointangleSplit.append(resample_timeseries(ja, t0, tnew))


    angleBetweenSegments = xdb.get_angle_between_segments(subject, trial,
                                                    ["LA", "LT"],
                                                    startTime=startTime,
                                                    anTime=anTime,
                                                    jointAxes=[],
                                                    useEKF=True,
                                                    doPlots=False)

    # Normalize by finding positive peak, setting angle to zero at negative peak, determining time vector
    # with zero at peak, and index of peak

    kneeangle_md = []
    tvec = np.linspace(0,99,100)
    #1/0
    for ja in jointangleSplit:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        if peakind < peakat+20 and peakind > peakat-20:
            t = tvec - peakind
        else:
            t = tvec - peakat
        # Look for valley in cycle before expected positive peak
        (negpeakind,) = np.nonzero(ja == np.min(ja[:peakat]))

        negpeakind = negpeakind[0]
        angle = ja - ja[negpeakind]
        kneeangle_md.append( (angle, t, peakind) )
        #angle = ja
        #t = tvec
        #kneeangle_md.append( (angle, t, 0) )

    kneeangle_imu = []
    for ja in angleBetweenSegments:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        if peakind < peakat+20 and peakind > peakat-20:
            t = tvec - peakind
        else:
            t = tvec - peakat
        (negpeakind,) = np.nonzero(ja == np.min(ja))
        negpeakind = negpeakind[0]
        angle = ja - ja[negpeakind]
        kneeangle_imu.append( (angle, t, peakind) )
        #angle = ja
        #t = tvec
        #kneeangle_imu.append( (angle, t, 0) )


    kneeangle_md_nooutliers = fix_outliers(kneeangle_md)
    kneeangle_imu_nooutliers = fix_outliers(kneeangle_imu)

    plt.figure()
    for ja in kneeangle_md_nooutliers:
        plt.plot(ja[1], ja[0]*180/np.pi)
    plt.title("Knee angle from marker data " + subject + ' trial ' + trial )
    plt.ylabel("Degrees")
    plt.savefig(resdir + '/' + subject + '_' + trial + '_knee-angle-mocap.pdf')

    #1/0
    plt.figure()
    for ja in kneeangle_imu_nooutliers:
        plt.plot(ja[1], ja[0]*180/np.pi)
    plt.title("Knee angle from imu data " + subject + ' trial ' + trial )
    plt.ylabel("Degrees")
    plt.savefig(resdir + '/' + subject + '_' + trial + '_knee-angle-imu.pdf')

    _stats_plot(kneeangle_md_nooutliers, kneeangle_imu_nooutliers, peakat)
    plt.title('Knee flexion. Ensemble mean +/- 2 std  ' + subject + ' trial ' + trial )
    plt.ylabel('Degrees')
    plt.xticks([])
    plt.savefig(resdir + '/' + subject + '_' + trial + '_knee-angle-mean-std.pdf')

def _stats_plot(dta_md, dta_imu, peakat=0):
    """
    Calculates ensemble mean and standard deviation from the sets of time series
    and plots.

    Arguments:
    dta_md  ->  list of timeseries obtained using markerdata. Each element
                corresponds to a gait cycles
    dta_imu ->  As above, but for imu data
    peakat  ->  Approximately procentage of cycle at which peak occurs. This is
                used in order to align the time series
    """

    imuflat = np.array([])
    mdflat = np.array([])
    imuvals = [np.array([]) for i in range(90)]
    mdvals = [np.array([]) for i in range(90)]
    tt = range(-70,20)

    #imumean = np.array([0, for i in range(100)])
    #imumean1 = np.array([0, for i in range(100)])
    #imustd = np.array([0, for i in range(100)])
    #mdmean = np.array([0, for i in range(100)])
    #mdmean1 = np.array([0, for i in range(100)])
    #mdstd = np.array([0, for i in range(100)])

    for (mda,imua) in itertools.izip(dta_md, dta_imu):
        mdta = mda[0]
        idta = imua[0]
        if (len(mdta) == 100) and (len(idta) == 100):
            mpind = mda[2]
            ipind = imua[2]
            if mpind < ipind:
                startoffset =  mpind-peakat
                endoffset = 99-ipind-(90-peakat)
            else:
                startoffset = ipind-peakat
                endoffset = 99-mpind-(90-peakat)

            imutvec = imua[1][ipind-peakat-startoffset:ipind+(90-peakat)+endoffset]
            mdtvec = mda[1][mpind-peakat-startoffset:mpind+(90-peakat)+endoffset]
            idta_ = idta[ipind-70-startoffset:ipind+(90-peakat)+endoffset]
            mdta_ = mdta[mpind-70-startoffset:mpind+(90-peakat)+endoffset]

            for (i_, val_)  in itertools.izip(imutvec,idta_):
                try:
                    itemind = tt.index( int(round(i_)) )
                    imuvals[itemind] = np.append(imuvals[itemind], val_)

                except ValueError:
                    pass


            for (i_, val_)  in itertools.izip(mdtvec,mdta_):
                try:
                    itemind = tt.index( int(round(i_)) )
                    mdvals[itemind] = np.append(mdvals[itemind], val_)

                except ValueError:
                    pass


            #for t in mda[1]:
            #    if t>-70 and t < 21
            imuflat = np.append(imuflat, idta_)
            mdflat = np.append(mdflat, mdta_)


    imumeans = np.array([np.mean(vals) for vals in imuvals])
    imustd = np.array([np.std(vals) for vals in imuvals])
    mdmeans = np.array([np.mean(vals) for vals in mdvals])
    mdstd = np.array([np.std(vals) for vals in mdvals])

    mygreen = (62.0/255, 151.0/255, 81.0/255)
    myblue = (57.0/255, 56.0/255, 187/255)
    #1/0
    plt.figure()
    pimu, = plt.plot(tt,imumeans*180/np.pi, linewidth=3, color=mygreen)
    plt.plot(tt,(imumeans+2*imustd)*180/np.pi, linewidth=1, linestyle='--', color=mygreen)
    plt.plot(tt,(imumeans-2*imustd)*180/np.pi, linewidth=1, linestyle='--', color=mygreen)
    pmd, = plt.plot(tt,mdmeans*180/np.pi, linewidth=3, color=myblue)
    plt.plot(tt,(mdmeans+2*mdstd)*180/np.pi, color=myblue, linewidth=1, linestyle='--')
    plt.plot(tt,(mdmeans-2*mdstd)*180/np.pi, color=myblue, linewidth=1, linestyle='--')

    plt.legend([pimu, pmd], ["IMU data", "Marker data"], loc=2)
    plt.xlabel('One gait cycle')



    #plt.figure()
    #bland_altman_plot(mdflat.ravel(), imuflat.ravel())
    #plt.title('Bland-Altman Plot')


def compare_foot_clearance(mocapdata, subject, trial,
                            resdir, peakat=0, startTime=60, anTime=180):
    """  Compares foot clearance calculated using IMU data to the vertical
    displacement of the ANKLE marker
    """

    (md, imudt) = get_comparison_data(mocapdata,
                                        ['ANKLE',],
                                        subject, trial, ['LA'],
                                        startTime, anTime)

    anklepos = md['ANKLE']
    ankleposz = anklepos.transpose()[:,2]

    # Split and normalize to 100 datapoints.
    ft = md['frametimes']
    cycledta = imudt['cycledata']

    (timeSplit, azSplitNoNorm) = split_in_cycles(ft, ankleposz, cycledta)
    tnew = np.linspace(0,100, 101)
    az_md = []
    for az_ in azSplitNoNorm:
        # Normalize to 101 points between 0 and 100
        azmin = np.min(az_)
        t0 = np.linspace(0,100, len(az_))
        az_md.append(resample_timeseries(az_-azmin, t0, tnew))


    [imuDisp, imuVel, imuGvec, cdta, cinds] = xdb.track_displacement(subject,
                                                trial, 'LA',
                                                startTime, anTime,
                                                doPlots = False)


    #az_imu = xdb.get_vertical_displacement(subject, trial, 'LA', startTime, anTime)

    az_imu = []
    for (d_, g_) in itertools.izip(imuDisp, imuGvec):
        vd = np.dot(d_[:,1:], g_)
        vDisps.append(np.max(vd) - np.min(vd))

    plt.figure()
    for az_ in az_md:
        plt.plot(az_[1], az_[0])
    plt.title("vertical displacement of ankle from marker data " + subject + ' trial ' + trial )
    plt.ylabel("m")
    plt.savefig(resdir + '/' + subject + '_' + trial + '_foot-clearance-mocap.pdf')

    plt.figure()
    for az_ in az_imu:
        plt.plot(az_[1], az_[0])

    plt.title("Vertical displacement of ankle from imu data " + subject + ' trial ' + trial )
    plt.ylabel("m")
    plt.savefig(resdir + '/' + subject + '_' + trial + '_foot-clearance-imu.pdf')

    _stats_plot(az_md, az_imu)
    plt.title('Foot clearance. Ensemble mean +/- 2 std  ' + subject + ' trial ' + trial )
    plt.ylabel('Degrees')
    plt.xticks([])
    plt.savefig(resdir + '/' + subject + '_' + trial + '_foot-clearance-mean-std.pdf')

    plt.show()

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

def markerdata_list():
    """
    Returns a list of markerdata file names
    """
    comparisonData = []
    dtaroot = '/media/ubuntu-15-10/home/kjartan/nvg/Data/'
    comparisonData.append( ("S4", "N", dtaroot + "S4/NVG_2012_S4_N.tsv" ) )
    comparisonData.append( ("S4", "D", dtaroot + "S4/NVG_2012_S4_D.tsv" ) )
    #comparisonData.append( ("S6", "N", dtaroot + "S6/NVG_2012_S6_N.tsv" ) )
    #comparisonData.append( ("S6", "D", dtaroot + "S6/NVG_2012_S6_D.tsv" ) )
    comparisonData.append( ("S10", "N", dtaroot + "S10/NVG_2012_S10_N.tsv" ) )
    comparisonData.append( ("S10", "D", dtaroot + "S10/NVG_2012_S10_D.tsv" ) )
    comparisonData.append( ("S12", "N", dtaroot + "S12/NVG_2012_S12_N.tsv" ) )
    comparisonData.append( ("S12", "D", dtaroot + "S12/NVG_2012_S12_D.tsv" ) )

    return comparisonData

def markerdata_sync_list():
    """
    Returns a list of markerdata filenames for the sync experiment
    """
    syncData = {}
    dtaroot = '/media/ubuntu-15-10/home/kjartan/nvg/Data/'
    syncData["S4"] =  dtaroot + "S4/NVG_2012_S4_sync.tsv"
    syncData["S6"] =  dtaroot + "S6/NVG_2012_S6_sync.tsv"
    syncData["S10"] =  dtaroot + "S10/NVG_2012_S10_sync.tsv"
    syncData["S12"] =  dtaroot + "S12/NVG_2012_S12_sync.tsv"

    return syncData


def plot_marker_data(markerdata = None, startTime=0, anTime=180, raw=True):
    """
    Will load the markerdata in the provided list and plot marker trajectories.
    @param comparisonData: list of comparison data to use. Each element in the
                           list is a tuple with (subj, trial, mcfilename)
    """

    if markerdata is None:
        md = markerdata_list()
    else:
        md = markerdata


    for (subject, trial, mcfilename) in md:

        [md, imudt] = get_comparison_data(mcfilename,
                                            ['ANKLE', 'KNEE', 'THIGH', 'HIP'],
                                            subject, trial, ['LA'],
                                            startTime, anTime)

        ankled = md['ANKLE']
        kneed = md['KNEE']
        thighd = md['THIGH']
        hipd = md['HIP']

        tvec = md['frametimes']
        cycledtaMC = md['cycledataMC']
        cycledtaSec = imudt['cycledataSec']

        #1/0
        plt.figure()
        k_ = 0
        for (m_, d_) in itertools.izip(["Ankle", "Knee", "Hip", "THigh"],
                                        [ankled, kneed, hipd, thighd]):
            k_ += 1
            plt.subplot(2,2,k_)
            plt.plot(tvec, d_.T)
            for (ic_, to_) in cycledtaSec:
                plt.plot([ic_, ic_], [-.3, 1], 'm', alpha=0.2)
                plt.plot([to_, to_], [-.4, .9], 'c', alpha=0.2)
            for (ic_, icn_) in cycledtaMC:
                plt.plot([tvec[ic_], tvec[ic_]], [-.3, 1], 'm')
                plt.plot([tvec[icn_], tvec[icn_]], [-.4, .9], 'c')
            plt.title(m_)

        plt.figure()
        plt.subplot(211)
        av = np.diff(ankled[2,:])*128.0
        plt.plot(tvec[:-1], av)
        for (ic_, to_) in cycledtaSec:
                plt.plot([ic_, ic_], [-1, 1.5], 'm', alpha=0.2)
                plt.plot([to_, to_], [-1.1, 1.4], 'c', alpha=0.2)

        for (ic_, icn_) in cycledtaMC:
            plt.plot([tvec[ic_], tvec[ic_]], [-1, 1.5], 'm')
            plt.plot([tvec[icn_], tvec[icn_]], [-1.1, 1.4], 'c')
        plt.title("Vertical velocity of ankle marker")

        plt.subplot(212)
        aa = np.diff(av)*128.0
        plt.plot(tvec[:-2], aa)
        for (ic_, to_) in cycledtaSec:
                plt.plot([ic_, ic_], [-40, 40], 'm', alpha=0.2)
                plt.plot([to_, to_], [-50, 30], 'c', alpha=0.2)
        for (ic_, icn_) in cycledtaMC:
            plt.plot([tvec[ic_], tvec[ic_]], [-40, 40], 'm')
            plt.plot([tvec[icn_], tvec[icn_]], [-50, 30], 'c')
        plt.title("Vertical acceleration of ankle marker")


def detect_heel_strike(tvec, ankleposz, wn=0.2, plotResults=False):
    """
    Returns a list of heelstrikes detected from the z-component of the ankle
    marker. From the ankle position, velocity and acceleration are computed.
    The following heuristic is used to detect the heel strike:
    position < 1cm above min
    velocity < 0
    acc detectpeaks

    Arguments:
    tvec        ->  Time vector (N,)
    ankleposz   ->  ankle position in vertical direction (N,)
    wn          ->  cutoff frequency of the low pass filter (Nyquist freq = 1)
    plotResults ->  If true do plot

    Returns:
    pks         <-  list of indices where peaks are found
    """

    # Lowpass filter using a Bessel filter
    [b,a] = signal.bessel(4, wn)
    ap = signal.filtfilt(b,a,ankleposz)

    dt = np.mean(np.diff(tvec))
    av = np.diff(ap)/dt
    aa = np.diff(av)/dt

    apmin = np.min(ap)

    okinds = np.where(np.logical_and(ap[:-2] < (apmin + 0.047),
                                      av[:-1] < 0.1))


    aaa = np.empty(aa.shape)
    aaa[:] = np.nan
    aaa[okinds] = aa[okinds]

    #pks = detect_peaks.detect_peaks(aa, mph=10, mpd=10)
    pks = detect_peaks.detect_peaks(aaa, mph=5, mpd=40)
    pks = np.intersect1d(pks, okinds)

    if plotResults:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(tvec, ankleposz, alpha=0.3)
        plt.plot(tvec, ap)
        for ic_ in pks:
            plt.plot([tvec[ic_], tvec[ic_]], [-0.3, 0], 'm', alpha=0.6)

        plt.subplot(3,1,2)
        plt.plot(tvec[:-1], av)
        for ic_ in pks:
            plt.plot([tvec[ic_], tvec[ic_]], [-1, 1], 'm', alpha=0.6)

        plt.subplot(3,1,3)
        plt.plot(tvec[:-2], aa)
        for ic_ in pks:
            plt.plot([tvec[ic_], tvec[ic_]], [-10, 10], 'm', alpha=0.6)

    return pks

def fix_outliers(timeseries):
    """
    Removes outliers from the set of timeseries. Outliers are identified as having
    positive and negative peak that are in the 5% or 95% percentile.

    Argument:
    timeseries -> list of tuples (dta, time, peakind)

    Returns:
    nts         <- Cleaned set of timeseries
    """

    posmax = [np.max(dt_) for (dt_, t_, p_) in timeseries]
    strt = [dt_[0] for (dt_, t_, p_) in timeseries]

    p05Max = np.percentile(posmax, 5)
    p95Max = np.percentile(posmax, 95)
    p05S = np.percentile(strt, 5)
    p95S = np.percentile(strt, 95)

    okindsMax = [k for k in range(len(posmax))
                            if (posmax[k]>p05Max and posmax[k]<p95Max) ]
    okindsStrt = [k for k in range(len(strt))
                            if (strt[k]>p05S and strt[k]<p95S) ]

    okinds = np.intersect1d(okindsStrt, okindsMax)
    return [timeseries[k] for k in okinds]



def main_2017(comparisonData = None, imu="LA", markers=["ANKLE", "KNEE"],
                                    startTime=60, anTime=20,
                                    var_angvel=1, var_incl=1e-2,
                                    lambda_gyro=1, lambda_incl=0.1,
                                    gThreshold=5e-1, resetAtIC=True,
                                    useCyclic=True):
    """
    Compares knee angle and foot clearance.
    @param comparisonData: list of comparison data to use. Each element in the
                           list is a tuple with (subj, trial, mcfilename)
    """
    resdir = "/home/kjartan/Dropbox/projekt/nvg/resultat/compare-mocap/" + date.isoformat(date.today())
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    if comparisonData is None:
        # Make list of all
        comparisonData = markerdata_list()


    _compare_angle_to_vertical(comparisonData, resdir,
                                imu=imu, markers=markers,
                                startTime=startTime,
                                anTime=anTime, var_angvel=var_angvel,
                                var_incl=var_incl, lambda_gyro=lambda_gyro,
                                lambda_incl=lambda_incl,
                                gThreshold=gThreshold,
                                resetAtIC=resetAtIC, useCyclic=useCyclic)

    #_compare_knee_angle(comparisonData, resdir)

    #_compare_foot_clearance(comparisonData, resdir)

def _compare_angle_to_vertical(comparisonData, resdir,
                                imu="LA", markers=["ANKLE", "KNEE"],
                                startTime=60, anTime=20,
                                var_angvel=1, var_incl=1e-2,
                                lambda_gyro=1, lambda_incl=0.1, gThreshold=5e-1,
                                resetAtIC=True, useCyclic=True):


    peakat = 70  # Approximately where in cycle peak appears
    for (subj, trial, mcfilename) in comparisonData:
        print("Subject %s trial %s" %(subj, trial))

        md = qtsv.loadQualisysTSVFile(mcfilename)
        compare_angle_to_vertical(md, subj,trial, resdir,
                                    imu=imu, markers=markers,
                                    startTime=startTime, anTime=anTime,
                                    peakat=peakat,
                                    resetAtIC=resetAtIC,
                                    var_angvel=var_angvel,
                                    var_incl=var_incl,
                                    lambda_gyro=lambda_gyro,
                                    lambda_incl=lambda_incl,
                                    gThreshold=gThreshold,
                                    useCyclic=useCyclic)


    plt.show()
def _compare_knee_angle(comparisonData, resdir):

    # Compare knee angle
    peakat = 70  # Approximately where in cycle peak appears
    for (subj, trial, mcfilename) in comparisonData:
        print("Subject %s trial %s" %(subj, trial))

        md = qtsv.loadQualisysTSVFile(mcfilename)
        compare_knee_RoM(md, subj,trial, resdir, peakat)


    plt.show()

def _compare_foot_clearance(comparisonData, resdir):

    #peakat = 70  # Approximately where in cycle peak appears
    for (subj, trial, mcfilename) in comparisonData:
        print("Subject %s trial %s" %(subj, trial))

        md = qtsv.loadQualisysTSVFile(mcfilename)
        compare_foot_clearance(md, subj,trial, resdir)


    plt.show()


def main_2016():
    resdir = "/home/kjartan/Dropbox/projekt/nvg/resultat/compare-mocap/" + date.isoformat(date.today())
    if not os.path.exists(resdir):
        os.makedirs(resdir)

    db = xdt.NVGData('/home/kjartan/Dropbox/Public/nvg201209.hdf5')
    #######
    # S7
    #######
    subj = "S7"
    peakat = 70  # Approximately where in cycle peak appears
    mcfilename = '/home/kjartan/Dropbox/projekt/nvg/data/solna09/S7/NVG_2012_S7_N.tsv'
    trial = "N"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, resdir, peakat)

def main():
    resdir = "/home/kjartan/Dropbox/projekt/nvg/resultat/compare-mocap/2014-08-08"

    db = xdt.NVGData('/home/kjartan/Dropbox/Public/nvg201209.hdf5')

    #######
    # S7
    #######


    #######
    # S4
    #######
    subj = "S4"
    peakat = 70  # Approximately where in cycle peak appears
    mcfilename = '/home/kjartan/nvg/Data/S4/NVG_2012_S4_N.tsv'
    trial = "N"
    #md = qtsv.loadQualisysTSVFile(mcfilename)
    #compare_knee_RoM(db, md, subj,trial, resdir, peakat)

    mcfilename = '/home/kjartan/nvg/Data/S4/NVG_2012_S4_D.tsv'
    trial = "D"
    #md = qtsv.loadQualisysTSVFile(mcfilename)
    #compare_knee_RoM(db, md, subj,trial, resdir, peakat)


    #######
    # S10
    #######
    subj = "S10"
    peakat = 70  # Approximately where in cycle peak appears
    mcfilename = '/home/kjartan/nvg/Data/S10/NVG_2012_S10_N.tsv'
    trial = "N"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, resdir, peakat)

    mcfilename = '/home/kjartan/nvg/Data/S10/NVG_2012_S10_D.tsv'
    trial = "D"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, resdir, peakat)


    #######
    # S12
    #######
    subj = "S12"
    peakat = 70  # Approximately where in cycle peak appears
    mcfilename = '/home/kjartan/nvg/Data/S12/NVG_2012_S12_N.tsv'
    trial = "N"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, resdir, peakat)

    mcfilename = '/home/kjartan/nvg/Data/S12/NVG_2012_S12_D.tsv'
    trial = "D"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, resdir, peakat)


    return db, md

def check_PN_vs_time(dateTimeFile=("/media/ubuntu-15-10/home/kjartan/nvg/"+
                                    "2012-09-19-S4/S4/LA-200/"+
                                    "NVG_2012_S4_A_LA_00203_DateTime.csv")):

    #dtapath = "/home/kjartan/Dropbox/projekt/nvg/data/test0910/6522/"
    #dtapath = "/home/kjartan/nvg/2012-09-20/S7/"
    #dtaset = "NVG_2012_S7_A_LT_00406";
    #tk = xdt.TimeKeeper(dtapath + dtaset + "_DateTime.csv")
    tk = xdt.TimeKeeper(dateTimeFile)

    t0 = tk.ttable[0][1]
    pns = np.array([row[0] for row in tk.ttable])
    times = np.array([(row[1]-t0).total_seconds() for row in tk.ttable])

    plt.figure()
    plt.plot(pns, times)
    plt.show()
    # OK! Straight line

    #Seconds per PN
    dt = (times[-1] - times[0]) / (pns[-1] - pns[0])
    print "dt = %0.4f seconds per packet number\n" % (dt,)
    print "samplefreq= %0.4f packets per second\n" % (1.0/dt,)
    print "But every second packet is missing in the data"

    return pns,times

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,        color='gray', linestyle='--')
    plt.axhline(md + 2*sd, color='gray', linestyle='--')
    plt.axhline(md - 2*sd, color='gray', linestyle='--')

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


if __name__ == '__main__':
    mdfiles = markerdata_list()
    plot_marker_data(mdfiles[:1])
    plt.show()
