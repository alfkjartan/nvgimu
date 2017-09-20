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

import gc
import os
import numpy.testing as npt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy import signal
import itertools
from datetime import datetime, timedelta, date
from nvg.io import qualisys_tsv as qtsv
from nvg.ximu import ximudata as xdt
from cyclicpython.algorithms import detect_peaks
from cyclicpython import cyclic_planar as cppl

# Load the imu data so that it is accessible as a global variable in this module
xdb = xdt.NVGData('/home/kjartan/Dropbox/Public/nvg201209.hdf5')

# Result directory is a global variable
RESDIR = ("/home/kjartan/Dropbox/projekt/nvg/resultat/compare-mocap/"
                                                + date.isoformat(date.today()) )
if not os.path.exists(RESDIR):
    os.makedirs(RESDIR)




def compare_angle_to_vertical(mocapdata, subject, trial="N",
                    startTime = 60,
                    anTime = 40,
                    imu="LA", markers=["ANKLE", "KNEE"],
                    nHarmonics=14,
                    peakat=70,
                    resetAtIC=False,
                    var_angvel=1,
                    var_incl=1e-2,
                    lambda_gyro=10,
                    lambda_incl=0.1,
                    gThreshold=5e-1,
                    useCyclic=True,
                    useLS=False,
                    plotResults=False,
                    startAtZero=False):
    """
    Compares angle to the vertical of given IMU and vector from first to second
    of the markers given.

    Arguments:
    mocapdata   -> Name of tsv file with markerdata, or markerdata object
    subject     -> Name of subject
    trial       -> Trial. Default "N"
    startTime   -> Where in trial to start comparison. Default 60 [seconds]
    anTime      -> Length of data sequence to compare. Default 40,
    imu         -> IMU analyze angle of Default "LA"
    markers     -> List of two markers defining the vertical direction of the
                    segment. Default ["ANKLE", "KNEE"]
    peakat      -> Approximately where in gait cycle the maximum angle is found
                   Default 70 [%]
    resetAtIC   -> If True, then computations restarts at each start of the gait
                   cycle. Default False
    var_angvel  -> Variance of the gyro data. Default 1 [rad^2/s^2]
    var_incl    -> Variance of the inclinometer data. Default 1e-2 [rad^2]
    lambda_gyro -> Weighting factor for gyro error in Robust QP optimization.
                   Default 10
    lambda_incl -> Weighting factor for inclinometer error. Default 0.1
    gThreshold  -> Threshold within which acceleration must be to be used for
                   inclinometer measurements. Default 5e-1 [m/s^2]
    useCyclic   -> If True, use the planar cyclic method. Otherwise integrate
                   gyro data.
    useLS       -> If True, use LS solver. Otherwise use robust QP solver.
                   Default False
    plotResults -> If True, generate plots. Default False
    startAtZero -> If True, results start at zero at start of gait cycle.
                   Default False

    Returns:
    tvec             <- Time vector (100,)
    meanIMU, stdIMU  <- Ensemble mean and std over gait cycles for IMU data
    meanMD, stdMD    <- Ensemble mean and std over gait cycles for marker data
    """

    (md, imudt) = xdb.get_comparison_data(mocapdata, markers,
                                subject, trial, [imu],
                                startTime, anTime)
    distd = md[markers[0]]
    proxd = md[markers[1]]


    ll_angle = kinematics.angle_to_vertical(upper=proxd, lower=distd)
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

    # Get the vertical reference direction
    vertRefDir = xdb.get_reference_vertical(subject, imu)

    print "Saggital direction from IMU"
    print d0
    print "Vertical reference direction from IMU"
    print vertRefDir

    a2vIMUSplit = []

    if useCyclic:
        if resetAtIC:
            omega=None
        else:
            cyclestartstop = np.asarray(cycledta)
            cyclePeriod = np.median(cyclestartstop[:,1] - cyclestartstop[:,0])
            omega = 2*np.pi/cyclePeriod
            print "Cycle period: %f" %cyclePeriod



        if useLS:
            solver = cppl.solve_lsq
        else:
            solver = cppl.solve_QP

        angleTracker = xdt.angle_to_vertical_cyclic_tracker(omega=omega,
                                                        nHarmonics=nHarmonics,
                                                        sagittalDir=d0,
                                                        vertRefDir=vertRefDir,
                                                        var_angvel=var_angvel,
                                                        var_incl=var_incl,
                                                        lambda_gyro=lambda_gyro,
                                                        lambda_incl=lambda_incl,
                                                        solver=solver)
    else:
        # if resetAtIC:
        #     m=None
        # else:
        #     m=50
        #
        # angleTracker = xdt.angle_to_vertical_ekf_tracker(sagittalDir=d0,
        #                                                 vertRefDir=vertRefDir,
        #                                                 var_angvel=var_angvel,
        #                                                 var_incl=var_incl,
        #                                                 m=m)

        angleTracker = xdt.angle_to_vertical_integrator_tracker(sagittalDir=d0,
                                                         vertRefDir=vertRefDir)

    a2vIMUSplitNoNorm = xdb.get_angle_to_vertical(subject, trial,
                                                imu,
                                                startTime=startTime,
                                                anTime=anTime,
                                                angleTracker=angleTracker,
                                                resetAtIC=resetAtIC,
                                                gThreshold=gThreshold,
                                                doPlots=plotResults)


    if plotResults:
        plt.figure()
        plt.plot(angleTracker.yinc[:,0], angleTracker.yinc[:,1], 'o')
        plt.plot(angleTracker.tvec, angleTracker.phi)

        plt.figure()
        plt.plot(angleTracker.tvec, angleTracker.gyrodta)

    for ja in a2vIMUSplitNoNorm:
        t0 = np.linspace(0,99, len(ja))
        a2vIMUSplit.append(resample_timeseries(ja, t0, tnew))



    angle_md = []
    tvec = np.linspace(0,99,100)
    #1/0
    for ja in angleSplit:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        if peakat is not None:
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
        else:
            angle_md.append( (ja, tvec, peakind) )


    angle_imu = []
    for ja in a2vIMUSplit:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        if peakat is not None:
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
        else:
            angle_imu.append( (ja, tvec, peakind) )

    angle_md_nooutliers = fix_outliers(angle_md)
    angle_imu_nooutliers = fix_outliers(angle_imu)

    if plotResults:
        plt.figure()
        for ja in angle_md_nooutliers:
            plt.plot(ja[1], ja[0]*180/np.pi)
        plt.title("Angle to vertical from marker data " + subject + ' trial ' + trial )
        plt.ylabel("Degrees")
        plt.savefig(RESDIR + '/' + subject + '_' + trial + '_' + imu
                        + '_angle_to_vertical_mocap.pdf')

        #1/0
        plt.figure()
        for ja in angle_imu_nooutliers:
            plt.plot(ja[1], ja[0]*180/np.pi)
        plt.title("Angle from imu data " + subject + ' trial ' + trial )
        plt.ylabel("Degrees")
        plt.savefig(RESDIR + '/' + subject + '_' + trial + '_' + imu
                    + '_angle_to_vertical_imu.pdf')


    (tvec, meanIMU, stdIMU, meanMD, stdMD) = _stats_plot(angle_md_nooutliers,
                                        angle_imu_nooutliers, peakat,
                                        startAtZero, plotResults)

    if plotResults:
        plt.title('Angle to vertical of lower leg. Ensemble mean +/- 2 std  ' + subject + ' trial ' + trial )
        plt.ylabel('Degrees')
        plt.xticks([])
        plt.savefig(RESDIR + '/' + subject + '_' + trial + '_' + imu
                    + '_angle_to_vertical-mean-std.pdf')


    return (tvec, meanIMU, stdIMU, meanMD, stdMD)

def compare_knee_RoM(mocapdata, subject, trial="N",
                    startTime = 60,
                    anTime = 240,
                    peakat=70):
    """  Compares Range of motion of knee flexion for trial N (Normal condition). """

    (md, imudt) = xdb.get_comparison_data(mocapdata,
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
    plt.savefig(RESDIR + '/' + subject + '_' + trial + '_knee-angle-mocap.pdf')

    #1/0
    plt.figure()
    for ja in kneeangle_imu_nooutliers:
        plt.plot(ja[1], ja[0]*180/np.pi)
    plt.title("Knee angle from imu data " + subject + ' trial ' + trial )
    plt.ylabel("Degrees")
    plt.savefig(RESDIR + '/' + subject + '_' + trial + '_knee-angle-imu.pdf')

    _stats_plot(kneeangle_md_nooutliers, kneeangle_imu_nooutliers, peakat)
    plt.title('Knee flexion. Ensemble mean +/- 2 std  ' + subject + ' trial ' + trial )
    plt.ylabel('Degrees')
    plt.xticks([])
    plt.savefig(RESDIR + '/' + subject + '_' + trial + '_knee-angle-mean-std.pdf')

def compare_sagittal_displacement(mocapdata, subject, trial="N",
                    startTime = 60,
                    anTime = 240,
                    imu="LA", marker="ANKLE",
                    nHarmonics = 14,
                    peakat=None,
                    resetAtIC=False,
                    var_angvel=1,
                    var_incl=1e-2,
                    var_acc=1,
                    lambda_gyro=10,
                    lambda_incl=0.1,
                    lambda_acc=1,
                    gThreshold=5e-1,
                    useCyclic=True,
                    useLS=False,
                    plotResults=False,
                    startAtZero=False):
    """  Compares foot clearance calculated using IMU data to the vertical
    displacement of the ANKLE marker

    Arguments:
    mocapdata   -> Name of tsv file with markerdata, or markerdata object
    subject     -> Name of subject
    trial       -> Trial. Default "N"
    startTime   -> Where in trial to start comparison. Default 60 [seconds]
    anTime      -> Length of data sequence to compare. Default 40,
    imu         -> IMU analyze angle of Default "LA"
    marker      -> Name of markers for which to compute the displacement.
                   Default "ANKLE"
    peakat      -> Approximately where in gait cycle the maximum angle is found
                   Default 0 [%]
    resetAtIC   -> If True, then computations restarts at each start of the gait
                   cycle. Default False
    var_angvel  -> Variance of the gyro data. Default 1 [rad^2/s^2]
    var_incl    -> Variance of the inclinometer data. Default 1e-2 [rad^2]
    lambda_gyro -> Weighting factor for gyro error in Robust QP optimization.
                   Default 10
    lambda_incl -> Weighting factor for inclinometer error. Default 0.1
    gThreshold  -> Threshold within which acceleration must be to be used for
                   inclinometer measurements. Default 5e-1 [m/s^2]
    useCyclic   -> If True, use the planar cyclic method. Otherwise EKF.
                   Default True
    useLS       -> If True, use LS solver. Otherwise use robust QP solver.
                   Default False
    plotResults -> If True, generate plots. Default False
    startAtZero -> If True, results start at zero at start of gait cycle.
                   Default False

    Returns:
    tvec             <- Time vector (100,)
    meanIMU, stdIMU  <- Ensemble mean and std over gait cycles for IMU data
    meanMD, stdMD    <- Ensemble mean and std over gait cycles for marker data
    """

    (md, imudt) = xdb.get_comparison_data(mocapdata, [marker],
                                        subject, trial, [imu],
                                        startTime, anTime)

    # Find the sagittal plane.
    posMD = md[marker]
    velMD = np.diff(posMD.T, axis=0)
    # These velocities lies in the sagittal plane
    (U,S,V) = np.linalg.svd(np.dot(velMD.T, velMD))
    sagittalDir = V[-1]

    # If not in direction of global negative x, then flip
    if sagittalDir[0] > 0:
        sagittalDir = -sagittalDir

    vertDir = np.array([0., 0., 1.])

    # Make horizontal and unit norm
    sagittalDir[2] = 0
    sagittalDir = sagittalDir / np.linalg.norm(sagittalDir)

    RGS = np.column_stack( (vertDir, np.cross(sagittalDir, vertDir), sagittalDir) )
    print RGS
    posMD = np.dot(RGS.T, md[marker]).T[:,:2]

    #1/0

    print "Saggital direction from marker data"
    print sagittalDir
    print "Vertical reference direction from IMU"
    print vertDir

    # Split and normalize to 100 datapoints.
    ft = md['frametimes']
    cycledtaMD =md['cycledataMC']


    (timeSplit, pSplitNoNorm) = split_in_cycles(ft, posMD,
                                                cycledtaMD, indices=True)

    print "--------------------------------------------"
    print "Number of cycles: %d" %len(pSplitNoNorm)
    print "--------------------------------------------"

    tnew = np.linspace(0,99, 100)
    pMDSplit= []
    for p_ in pSplitNoNorm:
        # Normalize to 100 points between 0 and 99
        if resetAtIC:
            p_ -= p_[0]

        t0 = np.linspace(0,99, len(p_))
        pMDSplit.append(resample_timeseries(p_, t0, tnew))

    tvec = np.linspace(0,99,100)

    pMD = []
    for p_ in pMDSplit:
        if peakat is not None:
            (peakind,) = np.nonzero(p_[:,0] == np.max(p_[:,0]))
            peakind = peakind[0]
            if peakind < peakat+20 and peakind > peakat-20:
                t = tvec - peakind
            else:
                t = tvec - peakat
        else:
            peakind = 0
            t = tvec

        pMD.append( (p_, t, peakind) )

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

    # Get the vertical reference direction
    vertRefDir = xdb.get_reference_vertical(subject, imu)

    print "Saggital direction from IMU"
    print d0
    print "Vertical reference direction from IMU"
    print vertRefDir


    if resetAtIC:
        omega=None
    else:
        cyclestartstop = np.asarray(cycledta)
        cyclePeriod = np.median(cyclestartstop[:,1] - cyclestartstop[:,0])
        omega = 2*np.pi/cyclePeriod
        print "Cycle period: %f" %cyclePeriod


    if useLS:
        solver = cppl.solve_lsq
    else:
        solver = cppl.solve_QP

    if useCyclic:
        displacementTracker = xdt.sagittal_plane_displacement_cyclic_tracker(
                                                            omega=omega,
                                                            nHarmonics=nHarmonics,
                                                            sagittalDir=d0,
                                                            vertRefDir=vertRefDir,
                                                            var_angvel=var_angvel,
                                                            var_incl=var_incl,
                                                            var_acc=var_acc,
                                                            lambda_gyro=lambda_gyro,
                                                            lambda_incl=lambda_incl,
                                                            lambda_acc=lambda_acc,
                                                            solver=solver)
    else:
        displacementTracker = xdt.sagittal_plane_displacement_integrator_tracker(
                                                        sagittalDir=d0,
                                                        vertRefDir=vertRefDir)

    pIMUSplitNoNorm = xdb.get_sagittal_plane_displacement(subject, trial,
                                            imu,
                                            startTime=startTime,
                                            anTime=anTime,
                                            displacementTracker=displacementTracker,
                                            resetAtIC=resetAtIC,
                                            gThreshold=gThreshold,
                                            doPlots=plotResults)


    pIMUSplit = []
    for p_ in pIMUSplitNoNorm:
        t0 = np.linspace(0,99, len(p_))
        pIMUSplit.append(resample_timeseries(p_, t0, tnew))


    tvec = np.linspace(0,99,100)

    pIMU = []
    for p_ in pIMUSplit:
        if peakat is not None:
            (peakind,) = np.nonzero(p_[:,0] == np.max(p_[:,0]))
            peakind = peakind[0]
            if peakind < peakat+20 and peakind > peakat-20:
                t = tvec - peakind
            else:
                t = tvec - peakat
        else:
            t = tvec
            peakind = 0
        pIMU.append( (p_, t, peakind) )


    #pMD_nooutliers0 = fix_outliers(pMD, index=0)
    #pMD_nooutliers = fix_outliers(pMD_nooutliers0, index=1)
    #pIMU_nooutliers0 = fix_outliers(pIMU, index=0)
    #pIMU_nooutliers = fix_outliers(pIMU_nooutliers0, index=1)

    pMD_nooutliers = pMD
    pIMU_nooutliers = pIMU

    if plotResults:
        plt.figure()
        for p_ in pMD_nooutliers:
            plt.subplot(121)
            plt.plot(p_[1], p_[0][:,0])
            plt.subplot(122)
            plt.plot(p_[1], p_[0][:,1])
        plt.subplot(121)
        plt.title("Displacement from marker data " + subject + ' trial ' + trial )
        plt.ylabel("x")
        plt.subplot(122)
        plt.ylabel("y")
        plt.savefig(RESDIR + '/' + subject + '_' + trial + '_' + imu
                        + '_displacement_mocap.pdf')

        #1/0
        plt.figure()
        for p_ in pIMU_nooutliers:
            plt.subplot(121)
            plt.plot(p_[1], p_[0][:,0])
            plt.subplot(122)
            plt.plot(p_[1], p_[0][:,1])
        plt.subplot(121)
        plt.title("Displacement from IMU data " + subject + ' trial ' + trial )
        plt.ylabel("x")
        plt.subplot(122)
        plt.ylabel("y")
        plt.savefig(RESDIR + '/' + subject + '_' + trial + '_' + imu
                        + '_displacement_imu.pdf')



    (tvec, xmeanIMU, xstdIMU, xmeanMD, xstdMD) = _stats_plot(
            [(p_[:,0], t_, i_) for (p_, t_, i_) in pMD_nooutliers],
            [(p_[:,0], t_, i_) for (p_, t_, i_) in pIMU_nooutliers],
            None, startAtZero, plotResults)
    if plotResults:
        plt.title('Displacement in sagittal plane. Ensemble mean +/- 2 std  '
                                                + subject + ' trial ' + trial )
        plt.ylabel('x')
        plt.xticks([])
        plt.savefig(RESDIR + '/' + subject + '_' + trial + '_' + imu
                    + '_displacement-x-mean-std.pdf')
    (tvec, ymeanIMU, ystdIMU, ymeanMD, ystdMD) = _stats_plot(
            [(p_[:,1], t_, i_) for (p_, t_, i_) in pMD_nooutliers],
            [(p_[:,1], t_, i_) for (p_, t_, i_) in pIMU_nooutliers],
            None, startAtZero, plotResults)
    if plotResults:
        plt.title('Displacement in sagittal plane. Ensemble mean +/- 2 std  '
                                                + subject + ' trial ' + trial )
        plt.ylabel('y')
        plt.xticks([])
        plt.savefig(RESDIR + '/' + subject + '_' + trial + '_' + imu
                    + '_displacement-y-mean-std.pdf')

    return (tvec, np.column_stack((xmeanIMU, ymeanIMU)),
                np.column_stack((xstdIMU, ystdIMU)),
                np.column_stack((xmeanMD, ymeanMD)),
                np.column_stack((xstdMD,ystdMD)))


    # az_imu = []
    # for (d_, g_) in itertools.izip(imuDisp, imuGvec):
    #     vd = np.dot(d_[:,1:], g_)
    #     vDisps.append(np.max(vd) - np.min(vd))
    #
    # plt.figure()
    # for az_ in az_md:
    #     plt.plot(az_[1], az_[0])
    # plt.title("vertical displacement of ankle from marker data " + subject + ' trial ' + trial )
    # plt.ylabel("m")
    # plt.savefig(RESDIR + '/' + subject + '_' + trial + '_foot-clearance-mocap.pdf')
    #
    # plt.figure()
    # for az_ in az_imu:
    #     plt.plot(az_[1], az_[0])
    #
    # plt.title("Vertical displacement of ankle from imu data " + subject + ' trial ' + trial )
    # plt.ylabel("m")
    # plt.savefig(RESDIR + '/' + subject + '_' + trial + '_foot-clearance-imu.pdf')
    #
    # _stats_plot(az_md, az_imu)
    # plt.title('Foot clearance. Ensemble mean +/- 2 std  ' + subject + ' trial ' + trial )
    # plt.ylabel('Degrees')
    # plt.xticks([])
    # plt.savefig(RESDIR + '/' + subject + '_' + trial + '_foot-clearance-mean-std.pdf')
    #
    # plt.show()


def markerdata_list():
    """
    Returns a list of markerdata file names
    """
    comparisonData = []
    dtaroot = '/media/ubuntu-15-10/home/kjartan/nvg/Data/'
    comparisonData.append( ("S4", "N", dtaroot + "S4/NVG_2012_S4_N.tsv" ) )
    comparisonData.append( ("S4", "D", dtaroot + "S4/NVG_2012_S4_D.tsv" ) )
    comparisonData.append( ("S6", "N", dtaroot + "S6/NVG_2012_S6_N.tsv" ) )
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

        [md, imudt] = xdb.get_comparison_data(mcfilename,
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

def _stats_plot(dta_md, dta_imu, peakat=None, startAtZero=False, plotResults=True):
    """
    Calculates ensemble mean and standard deviation from the sets of time series
    and plots.

    Arguments:
    dta_md  ->  list of timeseries obtained using markerdata. Each element
                corresponds to a gait cycles
    dta_imu ->  As above, but for imu data
    peakat  ->  Approximately procentage of cycle at which peak occurs. This is
                used in order to align the time series. If None, no alignment
    """

    imuflat = np.array([])
    mdflat = np.array([])
    if peakat is None:
        imuvals = [np.array([]) for i in range(100)]
        mdvals = [np.array([]) for i in range(100)]
        tt = range(100)
    else:
        imuvals = [np.array([]) for i in range(90)]
        mdvals = [np.array([]) for i in range(90)]
        tt = range(-peakat,100-peakat)

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

            if peakat is not None:
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

            else:
                imutvec = imua[1]
                mdtvec = mda[1]
                idta_ = idta
                mdta_ = mdta

            if startAtZero:
                idta_ -= idta_[0]
                mdta_ -= mdta_[0]

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


    if plotResults:
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

    return (tt, imumeans, imustd, mdmeans, mdstd)

    #plt.figure()
    #bland_altman_plot(mdflat.ravel(), imuflat.ravel())
    #plt.title('Bland-Altman Plot')



def fix_outliers(timeseries, index=None):
    """
    Removes outliers from the set of timeseries. Outliers are identified as having
    positive and negative peak that are in the 5% vel95% perc,
    ntile.

    Argument:
    timeseries -> list of tuples (dta, time, peakind)
    index      -> if the timeseries is 2D, then the element according to index
                  is analyzed. None means that the timeseries are scalar
    Returns:
    nts         <- Cleaned set of timeseries
    """

    if index is not None:
        timeseries = [(dt_[:,index], t_, p_) for (dt_, t_, p_) in timeseries]

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

def apply_cases(function, cases, args):
    """
    Runs the provided function for each element in kwlist.

    Arguments:
    function   ->  Function object or Callable
    cases      ->  Dictionary keyed by description, where each element is
                   a Dictionary of keyword arguments
    args       ->  Dictionary with arguments common for every call to function

    Returns:
    results    ->  Dictionary with results, indexed by the key in cases
    """

    results = {}
    for (case,kw) in cases.items():
        args_ = args.copy()
        args_.update(kw)
        print "Running function %s for case %s with arguments" %(
                                                            function.__name__,
                                                            case)
        print args_
        results[case] = function(**args_)

    return results




def main_2017(orientationComparison=True, displacementComparison=True,
                trialindices=None, **kwargs):
    """
    Main program to run several comparisons.
    kwargs can be used to override the default argument values in the
    dictionaries holding argument-value pairs.
    """
    comparisonData = markerdata_list() # Trials to use
    # List of (startTime, anTime)
    if trialindices is not None:
        comparisonData = [comparisonData[i] for i in trialindices]

    dataset = [ (30,60), (90,60), (150, 60), (210, 60), (270, 60) ]

    if orientationComparison:
        angle2vertArgs = {"imu":"LA", "markers":["ANKLE", "KNEE"],
                              "startTime":60, "anTime":20,
                              "nHarmonics":16,
                              "gThreshold":5e-1, "plotResults":False,
                              "startAtZero":False,
                              "peakat":None}
        angle2vertArgs.update(kwargs)

        angle2vertCases = { "Fourier-series": {"useCyclic":True, "resetAtIC":True,
                                                "useLS":False,
                                                "lambda_gyro":1, "lambda_incl":0.1,
                                                "var_angvel":1, "var_incl":1},
                                #"Cyclic LS": {"useCyclic":True, "resetAtIC":True,
                                #                "useLS":True,
                                #                "var_angvel":1, "var_incl":1},

                                "Integrate": { "useCyclic":False,
                                            "var_angvel":10, "var_incl":1}}


        for (subj, trial, mcfilename) in comparisonData:
            angle2vertArgs["mocapdata"] = mcfilename
            angle2vertArgs["subject"] = subj
            angle2vertArgs["trial"] = trial

            for (startTime, anTime) in dataset:
                angle2vertArgs["startTime"] = startTime
                angle2vertArgs["anTime"] = anTime
                resultat = apply_cases(compare_angle_to_vertical,
                                        angle2vertCases,
                                        angle2vertArgs)


                _plot_cases_results(resultat, subj, trial,
                            '%d-%d' %(startTime, startTime+anTime),
                            "angle2vertical")

                gc.collect()
    if displacementComparison:
        displacementArgs = {"imu":"LA", "marker":"ANKLE",
                              "startTime":60, "anTime":20,
                              "nHarmonics":24,
                              "var_angvel":1, "var_incl":1,
                              "var_acc":1,
                              "lambda_gyro":1, "lambda_incl":0.1,
                              "lambda_acc":1,
                              "gThreshold":5e-1, "plotResults":False,
                              "startAtZero":False}
        displacementArgs.update(kwargs)

        displacementCases = { "Fourier-series": {"resetAtIC":True,
                                                    "useLS":False},
                              "Integrate": {"resetAtIC":True,
                                                "useCyclic":False}}


        for (subj, trial, mcfilename) in comparisonData:
            displacementArgs["mocapdata"] = mcfilename
            displacementArgs["subject"] = subj
            displacementArgs["trial"] = trial

            for (startTime, anTime) in dataset:
                displacementArgs["startTime"] = startTime
                displacementArgs["anTime"] = anTime
                resultat = apply_cases(compare_sagittal_displacement,
                                    displacementCases,
                                    displacementArgs)


                _plot_cases_results(resultat, subj, trial,
                            '%d-%d' %(startTime, startTime+anTime),
                            "displacement")
                gc.collect()


def _plot_cases_results(resultat, subj, trial, dataset, test):
    #mpl.style.use('fivethirtyeight')
    mpl.style.use('seaborn-deep')
    legends=[]

    (case_, res_) = resultat.iteritems().next()
    (tvec, imuMean, imuStd, mdMean, mdStd) = res_

    isScalarSeries = False
    if mdMean.ndim == 1:
        isScalarSeries = True

    if isScalarSeries:
        fig=plt.figure(figsize=(6,4))
        plt.plot(tvec, mdMean*180.0/np.pi, linewidth=3)
        legends.append("Marker data")
        for (case_, res_) in resultat.items():
            (tvec, imuMean, imuStd, mdMean, mdStd) = res_
            plt.plot(tvec, imuMean*180.0/np.pi, linewidth=3)
            legends.append(case_)

        plt.gca().set_color_cycle(None)
        (case_, res_) = resultat.iteritems().next()
        (tvec, imuMean, imuStd, mdMean, mdStd) = res_
        plt.plot(tvec, (mdMean+2*mdStd)*180.0/np.pi, linewidth=3,
                                    linestyle='--')
        for (case_, res_) in resultat.items():
            (tvec, imuMean, imuStd, mdMean, mdStd) = res_
            plt.plot(tvec, (imuMean+2*imuStd)*180.0/np.pi, linewidth=1,
                                                                linestyle='--')
        plt.gca().set_color_cycle(None)
        (case_, res_) = resultat.iteritems().next()
        (tvec, imuMean, imuStd, mdMean, mdStd) = res_
        plt.plot(tvec, (mdMean-2*mdStd)*180.0/np.pi, linewidth=3,
                                    linestyle='--')
        for (case_, res_) in resultat.items():
            (tvec, imuMean, imuStd, mdMean, mdStd) = res_
            plt.plot(tvec, (imuMean-2*imuStd)*180.0/np.pi, linewidth=1,
                                                                linestyle='--')
        plt.ylabel('Degrees')
    else:
        fig = plt.figure(figsize=(6,8))
        for i_ in range(2):
            plt.subplot(2, 1, i_+1)

            plt.plot(tvec, mdMean[:,i_], linewidth=3)
            if i_==0:
                legends.append("Marker data")
            for (case_, res_) in resultat.items():
                (tvec, imuMean, imuStd, mdMean, mdStd) = res_
                plt.plot(tvec, imuMean[:,i_], linewidth=3)
                if i_==0:
                    legends.append(case_)

            plt.gca().set_color_cycle(None)
            (case_, res_) = resultat.iteritems().next()
            (tvec, imuMean, imuStd, mdMean, mdStd) = res_
            plt.plot(tvec, (mdMean[:,i_]+2*mdStd[:,i_]),
                                        linewidth=3, linestyle='--')
            for (case_, res_) in resultat.items():
                (tvec, imuMean, imuStd, mdMean, mdStd) = res_
                plt.plot(tvec, (imuMean[:,i_]+2*imuStd[:,i_]),
                                        linewidth=1, linestyle='--')
            plt.gca().set_color_cycle(None)
            (case_, res_) = resultat.iteritems().next()
            (tvec, imuMean, imuStd, mdMean, mdStd) = res_
            plt.plot(tvec, (mdMean[:,i_]-2*mdStd[:,i_]),
                                        linewidth=3, linestyle='--')
            for (case_, res_) in resultat.items():
                (tvec, imuMean, imuStd, mdMean, mdStd) = res_
                plt.plot(tvec, (imuMean[:,i_]-2*imuStd[:,i_]),
                                        linewidth=1, linestyle='--')

            plt.ylabel('meter')


    plt.legend(legends, loc=0)
    plt.xlabel('Gait cycle (%)')

    plt.savefig(RESDIR + '/' + test + '_' + subj + '_' + trial + '_' + dataset + '.pdf')
    plt.close(fig)
    plt.close("all")


def do_compare_angle_to_vertical(comparisonData,
                                imu="LA", markers=["ANKLE", "KNEE"],
                                startTime=60, anTime=20,
                                var_angvel=1, var_incl=1e-2,
                                lambda_gyro=1, lambda_incl=0.1, gThreshold=5e-1,
                                resetAtIC=True, useCyclic=True, useLS=False):


    peakat = 70  # Approximately where in cycle peak appears
    for (subj, trial, mcfilename) in comparisonData:
        print("Subject %s trial %s" %(subj, trial))

        md = qtsv.loadQualisysTSVFile(mcfilename)
        compare_angle_to_vertical(md, subj,trial,
                                    imu=imu, markers=markers,
                                    startTime=startTime, anTime=anTime,
                                    peakat=peakat,
                                    resetAtIC=resetAtIC,
                                    var_angvel=var_angvel,
                                    var_incl=var_incl,
                                    lambda_gyro=lambda_gyro,
                                    lambda_incl=lambda_incl,
                                    gThreshold=gThreshold,
                                    useCyclic=useCyclic,
                                    useLS=useLS)


    plt.show()
def _compare_knee_angle(comparisonData):

    # Compare knee angle
    peakat = 70  # Approximately where in cycle peak appears
    for (subj, trial, mcfilename) in comparisonData:
        print("Subject %s trial %s" %(subj, trial))

        md = qtsv.loadQualisysTSVFile(mcfilename)
        compare_knee_RoM(md, subj,trial, peakat)


    plt.show()

def _compare_foot_clearance(comparisonData):

    #peakat = 70  # Approximately where in cycle peak appears
    for (subj, trial, mcfilename) in comparisonData:
        print("Subject %s trial %s" %(subj, trial))

        md = qtsv.loadQualisysTSVFile(mcfilename)
        compare_foot_clearance(md, subj,trial)


    plt.show()


def main_2016():

    db = xdt.NVGData('/home/kjartan/Dropbox/Public/nvg201209.hdf5')
    #######
    # S7
    #######
    subj = "S7"
    peakat = 70  # Approximately where in cycle peak appears
    mcfilename = '/home/kjartan/Dropbox/projekt/nvg/data/solna09/S7/NVG_2012_S7_N.tsv'
    trial = "N"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, peakat)

def main():
    RESDIR = "/home/kjartan/Dropbox/projekt/nvg/resultat/compare-mocap/2014-08-08"

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
    #compare_knee_RoM(db, md, subj,trial, RESDIR, peakat)

    mcfilename = '/home/kjartan/nvg/Data/S4/NVG_2012_S4_D.tsv'
    trial = "D"
    #md = qtsv.loadQualisysTSVFile(mcfilename)
    #compare_knee_RoM(db, md, subj,trial, RESDIR, peakat)


    #######
    # S10
    #######
    subj = "S10"
    peakat = 70  # Approximately where in cycle peak appears
    mcfilename = '/home/kjartan/nvg/Data/S10/NVG_2012_S10_N.tsv'
    trial = "N"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, peakat)

    mcfilename = '/home/kjartan/nvg/Data/S10/NVG_2012_S10_D.tsv'
    trial = "D"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, peakat)


    #######
    # S12
    #######
    subj = "S12"
    peakat = 70  # Approximately where in cycle peak appears
    mcfilename = '/home/kjartan/nvg/Data/S12/NVG_2012_S12_N.tsv'
    trial = "N"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, peakat)

    mcfilename = '/home/kjartan/nvg/Data/S12/NVG_2012_S12_D.tsv'
    trial = "D"
    md = qtsv.loadQualisysTSVFile(mcfilename)
    compare_knee_RoM(db, md, subj,trial, peakat)


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

def test_detect_heel_strike(listindex=[0]):
        mdfiles = markerdata_list()
        for ind in listindex:
            (subj, trial, mcfilename) = mdfiles[ind]
            (md, imudt) = xdb.get_comparison_data(mcfilename, ["ANKLE"],
                                    subj, trial, ["LA"], plotResults=True)

if __name__ == '__main__':
    mdfiles = markerdata_list()
    plot_marker_data(mdfiles[:1])
    plt.show()
