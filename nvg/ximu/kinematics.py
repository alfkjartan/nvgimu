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


#-------------------------------------------------------------------------------
# Callable classes that estimate orientation. It is assumed that the data
# provided is for one single gait cycle
# the following arguments:
#    tvec   ->  A numpy (N,) array of time stamps for each data point
#    gyro   ->  A numpy (N,3) array of gyro data in [rad/s]
#    acc    ->  A numpy (N,3) array of accelerations [m/s^2]
#    mag    ->  A numpy (N,3) array of magnetometer data
#
# Returns
#    qEst    <- List of QuaternionArrays. One item per cycle
#
# Any other parameters the algorithm depends upon is set during instantiation
#-------------------------------------------------------------------------------

class CyclicEstimator:
    def __init__(self, nHarmonics, detrendData=True, doPlots = False):
        self.nHarmonics = nHarmonics
        self.detrendData = detrendData
        self.doPlots = doPlots

    def estimate(self, imudta, doPlots):
        """
        DEPRECATED. Use callable __call__ instead
        Runs the cyclic orientation method assuming that the imud is a single cycledta
        """
        dt = 1.0/256.0
        tvec = imudta[:,0]*dt

        #accdta = imudta[:,4:7]*9.82
        gyrodta = imudta[:,1:4]*np.pi/180.0
        magdta = imudta[:,7:10]

        omega = 2*np.pi/ (tvec[-1]  - tvec[0])

        (qEst, bEst) = cyclic_path.estimate_cyclic_orientation(tvec, gyrodta,
                        magdta, omega, self.nHarmonics)
        tvec.shape = (len(tvec), 1)
        return np.hstack((tvec, qEst))

    def __call__(self, tvec, gyro, acc, mag,
                            gyroref=None, accref=None, magref=None):
        omega = 2*np.pi/ (tvec[-1]  - tvec[0])
        if self.detrendData:
            w = detrend(gyro, type='constant', axis=0)
        else:
            w = gyro

        (qE, bE) = cyclic_path.estimate_cyclic_orientation(
                                        tvec, w, mag, omega, self.nHarmonics)

        q = quat.QuaternionArray(qE)
        if (accref is not None) and (magref is not None):
            phi = angle_to_accref(q, acc, accref, gyro, magref)
        else:
            phi = None
        return (q, phi)

class GyroIntegratorOrientation:
    def __init__(self, detrendData=True, doPlots = False):
        self.doPlots = doPlots
        self.detrendData = detrendData

    def estimate(self, imudta, doPlots):
        """
        DEPRECATED. Use callable __call__ instead
        Runs the cyclic orientation method assuming that the imud is a single cycledta
        """
        imuq = np.zeros((imudta.shape[0], 5))
        initRotation = quat.Quaternion(1,0,0,0)
        gMagn = 9.82
        deg2rad = np.pi/180.0
        imudtaT = imudta.T
        t = imudta[0,0]*self.dt
        orientFilter = None
        for i in range(0,imuq.shape[0]):
            if i == 0:
                orientFilter = orientation.GyroIntegrator(t, initRotation)
            else:
                t = imudta[i,0]*self.dt
                orientFilter(imudtaT[4:7,i]*gMagn, imudtaT[7:10,i],
                        imudtaT[1:4,i]*deg2rad, t)

            imuq[i,0] = t
            imuq[i,1] = orientFilter.rotation.latestValue.w
            imuq[i,2] = orientFilter.rotation.latestValue.x
            imuq[i,3] = orientFilter.rotation.latestValue.y
            imuq[i,4] = orientFilter.rotation.latestValue.z


        if doPlots: # Check results
            pyplot.figure()
            pyplot.plot(imuq[:,0], imuq[:,1:5])

        #return [imuq, cycledtainds]
        return imuq

    def __call__(self, tvec, gyro, acc, mag,
                                    gyroref=None, accref=None, magref=None):

        if self.detrendData:
            w = detrend(gyro, type='constant', axis=0)
        else:
            w = gyro

        initRotation = quat.Quaternion(1,0,0,0)
        orientFilter = orientation.GyroIntegrator(tvec[0], initRotation)
        for (t_, a_, m_, g_) in itertools.izip(tvec, acc, mag, w):
                orientFilter(a_, mag_, gyro_, t_)

        q = orientFilter.rotation.values()

        if (accref is not None) and (magref is not None):
            phi = angle_to_accref(q, acc, accref, gyro, magref)
        else:
            phi = None
        return (q, phi)

class EKFOrientation:
    def estimate(self, imudta, doPlots):
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

        #return [imuq, cycledtainds]
        return imuq

class angle_to_vertical_integrator_tracker(object):
    """
    Callable that tracks the angle to vertical of a segment by integrating
    the angular velocity projected onto the sagittal plane.
    """

    def __init__(self, sagittalDir, vertRefDir=np.array([-1.0, 0, 0]),
                        g=9.82, gThreshold=1e-1, plotResults=False):
        self.sagittalDir = sagittalDir
        self.vertRefDir = vertRefDir
        self.g = g
        self.gThreshold = gThreshold
        self.plotResults = plotResults
        self.tvec = None
        self.yinc = None
        self.phi = None
        self.gyrodta = None
        self.accdta = None

    def __call__(self, tvec, acc, gyro, mag):

        # Get the inclination measurements
        horRef = np.cross(self.sagittalDir, self.vertRefDir)
        (tauK, yinc) = inclination(acc, self.vertRefDir, horRef,
                                                self.g, self.gThreshold)
                                                # Integrate the projected zero-mean gyro data

        w = detrend(np.dot(gyro, self.sagittalDir), type='constant')
        wint = cumtrapz(w, tvec)
        wint = np.insert(wint, 0, wint[0])
        # translate the angle (add constant offset) to best match inclination
        # measurements
        # phi[tauK] = wint[tauK] + offset = yinc
        # 1*offset = yinc - wint[tauK]
        # offset = 1' * (yinc - wint[tauK])/len(tauK)
        phi = wint + np.mean(yinc - wint[tauK])

        self.phi = phi
        self.yinc = np.column_stack( (tvec[tauK], yinc))
        self.tvec = tvec
        self.gyrodta = gyro
        self.accdta = acc

        # Return a Quaternion array
        return [quat.QuaternionFromAxis(self.sagittalDir, phi_) for phi_ in self.phi]


class angle_to_vertical_ekf_tracker(object):
    """
    Callable that tracks the angle to vertical of a segment using a
    fixed-lag EKF.
    """

    def __init__(self, sagittalDir, vertRefDir=np.array([-1.0, 0, 0]),
                                    var_angvel=1e-2, var_incl=1e-1, m=None,
                                    g = 9.82, gThreshold=1e-1,
                                    plotResults = False):
        self.sagittalDir = sagittalDir
        self.vertRefDir = vertRefDir
        self.var_angvel = var_angvel
        self.var_incl = var_incl
        self.m = m
        self.g = g
        self.gThreshold = gThreshold
        self.plotResults = plotResults
        self.tvec = None
        self.yinc = None
        self.phi = None
        self.gyrodta = None
        self.accdta = None

    def __call__(self, tvec, acc, gyro):

        (phi, yincl) = cpekf.track_planar_vertical_orientation(tvec,
                                                    acc, gyro,
                                                    self.sagittalDir,
                                                    self.var_angvel,
                                                    self.var_incl,
                                                    self.m,
                                                    vertRefDir=self.vertRefDir,
                                                    g=self.g,
                                                    gThreshold=self.gThreshold,
                                                    plotResults=self.plotResults)
        self.phi = phi
        self.yinc = yincl
        self.tvec = tvec
        self.gyrodta = gyro
        self.accdta = acc

        return [quat.QuaternionFromAxis(self.sagittalDir, phi_) for phi_ in self.phi]


class angle_to_vertical_cyclic_tracker(object):
    """
    Callable that tracks the angle to vertical of a segment using the planar
    cyclic method.

    The orienation is defined by the single angle phi, which is defined as

    """

    def __init__(self, omega, nHarmonics,
                        sagittalDir, vertRefDir=np.array([-1.0, 0, 0]),
                        var_angvel=1, var_incl=1e-1,
                        lambda_gyro=1, lambda_incl=0.1,
                        solver=cppl.solve_QP, g=9.82, gThreshold=1e-1,
                        plotResults = False):
        self.sagittalDir=sagittalDir
        self.vertRefDir=vertRefDir
        self.omega = omega
        self.nHarmonics = nHarmonics
        self.solver=solver
        self.lambda_gyro = lambda_gyro
        self.lambda_incl = lambda_incl
        self.var_gyro = var_angvel
        self.var_incl = var_incl
        self.g = g
        self.gThreshold = gThreshold
        self.plotResults = plotResults
        self.link = None
        self.tvec = None
        self.yinc = None
        self.phi = None
        self.gyrodta = None
        self.accdta = None



    def __call__(self, tvec, acc, gyro):
        link = cppl.Link(tvec, acc, gyro, self.sagittalDir, self.vertRefDir)
        if self.omega is None:
            # One cycle in data
            T = tvec[-1] - tvec[0]
            omega = 2*np.pi/T
        else:
            omega = self.omega
        link.estimate_planar_cyclic_orientation(omega, self.nHarmonics,
                                            g=self.g,
                                            gThreshold=self.gThreshold,
                                            var_gyro=self.var_gyro,
                                            var_incl=self.var_incl,
                                            lambda_gyro=self.lambda_gyro,
                                            lambda_incl=self.lambda_incl,
                                            solver=self.solver)

        self.link = link
        self.phi = link.phi
        self.yinc = link.yinc
        self.tvec = link.tvec
        self.gyrodta = link.gyrodta
        self.accdta = link.accdta
        return [quat.QuaternionFromAxis(self.sagittalDir, phi_) for phi_ in self.phi]

#-------------------------------------------------------------------------------
# Callable classes that estimate displacement. It is assumed that the data
# provided is for one single gait cycle
# the following arguments:
#    tvec   ->  A numpy (N,) array of time stamps for each data point
#    acc    ->  A numpy (N,3) array of accelerations [m/s^2]
#    qq     ->  A QuaternionArray with estiamted orientation of the IMU
#
# Returns
#    d, v, g    <- Numpy arraus (N,3) with displacement, velocity and g_vector
#                  The displacement is in the reference frame that is rotated by
#                  qq.
#
# Any other parameters the algorithm depends upon is set during instantiation
#-------------------------------------------------------------------------------

class IntegrateAccelerationDisplacementEstimator:

    def __call__(self, tvec, acc, qq):
        # Rotate acceleration measurements, then remove linear trend.
        # Rotate acceleration vectors
        acc_S = qq.rotateVector(acc.T).T
        # Since the IMU returns to the same position, the average acceleration
        # must be one g pointing upwards.
        g = np.mean(acc_S, axis=0)
        gN = g / np.linalg.norm(g) # Normalize

        acc_S_detrend = detrend(acc_S, type='linear', axis=0)

        vel = cumtrapz(acc_S_detrend, tvec, axis=0)
        vel = np.insert(vel, 0, vel[0], axis=0)
        disp = cumtrapz(vel, tvec, axis=0)
        disp = np.insert(disp, 0, disp[0], axis=0)
        return (disp, vel, gN)



class CyclicPlanarDisplacementEstimator:
    def __init__(self, nHarmonics):
        self.nHarmonics = nHarmonics

    def estimate(self, imudta, sagittalDir, doPlots):
        """
        Runs the cyclic planar displacement method assuming that the
        imud is a single cycledta
        """
        dt = 1.0/256.0
        tvec = imudta[:,0]*dt

        #accdta = imudta[:,4:7]*9.82
        gyrodta = imudta[:,1:4]*np.pi/180.0
        magdta = imudta[:,7:10]

        omega = 2*np.pi/ (tvec[-1]  - tvec[0])

        (qEst, bEst) = cyclic_path.estimate_cyclic_orientation(tvec, gyrodta,
                        magdta, omega, self.nHarmonics)
        tvec.shape = (len(tvec), 1)
        return np.hstack((tvec, qEst))




class sagittal_plane_displacement_integrator_tracker(object):
    """
    Callable that tracks the displacement in the sagittal plane by
    integrating the acceleration twice using the cumtrapz function
    """

    def __init__(self, sagittalDir, vertRefDir=np.array([-1.0, 0, 0]),
                    g=9.82, gThreshold=1e-1, plotResults=False):
        self.sagittalDir=sagittalDir
        self.vertRefDir=vertRefDir
        self.g = g
        self.gThreshold = gThreshold
        self.plotResults = plotResults
        self.tvec = None
        self.yinc = None
        self.phi = None
        self.gyrodta = None
        self.accdta = None

        self.angleTracker = angle_to_vertical_integrator_tracker(sagittalDir,
                                                                vertRefDir,
                                                                g, gThreshold,
                                                                plotResults)

    def __call__(self, tvec, acc, qE):

        #qE = self.angleTracker(tvec, acc, gyro)
        phi = self.angleTracker.phi

        RLG = cppl.R_LG(phi, self.sagittalDir, self.vertRefDir)
        accPlanar = cppl.rotate_vectors(RLG, acc, transpose=True)
        accPlanar -= np.mean(accPlanar, axis=0)
        velPlanar = cumtrapz(accPlanar, tvec, axis=0)
        velPlanar = np.reshape(np.insert(velPlanar, 0, velPlanar[0]), (len(tvec), 3))
        velPlanar -= np.mean(velPlanar, axis=0)
        dispPlanar = cumtrapz(velPlanar, tvec, axis=0)
        dispPlanar = np.reshape(np.insert(dispPlanar, 0, dispPlanar[0]), (len(tvec), 3))

        self.phi = phi
        self.tvec = tvec
        self.gyrodta = gyro
        self.accdta = acc

        # Calculate displacement in 3D using the direction of vertical and forward

        return (dispPlanar, acc, accPlanar)

class sagittal_plane_displacement_cyclic_tracker(object):
    """
    Callable that tracks the displacement in the sagittal plane using the planar
    cyclic method.
    """

    def __init__(self, omega, nHarmonics,
                    sagittalDir, vertRefDir=np.array([-1.0, 0, 0]),
                    var_angvel=1, var_incl=1e-1, var_acc=1,
                    lambda_gyro=1, lambda_incl=0.1, lambda_acc=1,
                    solver=cppl.solve_QP):
        self.sagittalDir=sagittalDir
        self.vertRefDir=vertRefDir
        self.omega = omega
        self.nHarmonics = nHarmonics
        self.solver=solver
        self.lambda_gyro = lambda_gyro
        self.lambda_incl = lambda_incl
        self.lambda_acc = lambda_acc
        self.var_gyro = var_angvel
        self.var_incl = var_incl
        self.var_acc = var_acc
        self.link = None
        self.tvec = None
        self.yinc = None
        self.phi = None
        self.gyrodta = None
        self.accdta = None



    def __call__(self, tvec, acc, gyro, g=9.82, gThreshold=1e-1,
             plotResults=False):
        link = cppl.Link(tvec, acc, gyro, self.sagittalDir, self.vertRefDir)
        if self.omega is None:
            # One cycle in data
            T = tvec[-1] - tvec[0]
            omega = 2*np.pi/T
        else:
            omega = self.omega
        link.estimate_planar_cyclic_orientation(omega, self.nHarmonics,
                                            g=g, gThreshold=gThreshold,
                                            var_gyro=self.var_gyro,
                                            var_incl=self.var_incl,
                                            lambda_gyro=self.lambda_gyro,
                                            lambda_incl=self.lambda_incl,
                                            solver=self.solver,
                                            plotResults=plotResults)

        link.estimate_planar_cyclic_displacement( self.nHarmonics,  g=g,
                                                lambd=self.lambda_acc,
                                                solver=self.solver)


        self.link = link
        self.phi = link.phi
        self.yinc = link.yinc
        self.tvec = link.tvec
        self.gyrodta = link.gyrodta


        if plotResults:
            acc_G = link.accdtaSagittal # Zero-mean acc data

            pyplot.figure()
            pyplot.subplot(211)
            pyplot.plot(link.tvec, acc_G[:,0])
            pyplot.plot(link.tvec[[0, -1]],
                                    np.mean(acc_G[:,0])*np.array([1,1]))
            pyplot.plot(link.tvec, link.acc[:,0], linewidth=2)
            pyplot.plot(link.tvec[[0, -1]],
                                    np.mean(link.acc[:,0])*np.array([1,1]),
                                    linewidth=2)
            pyplot.title("Acceleration in vertical direction")
            pyplot.subplot(212)
            pyplot.plot(link.tvec, acc_G[:,1])
            pyplot.plot(link.tvec[[0, -1]],
                                    np.mean(acc_G[:,1])*np.array([1,1]))
            pyplot.plot(link.tvec, link.acc[:,1], linewidth=2)
            pyplot.plot(link.tvec[[0, -1]],
                                    np.mean(link.acc[:,1])*np.array([1,1]),
                                    linewidth=2)
            pyplot.title("Acceleration in horizontal direction")
            #1/0

        return (link.disp, link.acc, link.accdtaSagittal)




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

    vert = np.asarray(vertDir)

    vec = upper - lower
    # Normalize the vectors
    norms = np.apply_along_axis(np.linalg.norm, 1, vec )
    normVecT = vec.T/norms # Will be (3,N)
    vec = normVecT.T

    if sagittalDir is None:
        #costheta = np.dot(vert, normVecT)
        #return np.arccos(costheta)

        # Find the sagittal plane.
        vecVel = np.diff(vec, axis=0)
    # These velocities lies in the sagittal plane
        vecVel = vecVel[~np.isnan(np.sum(vecVel, axis=1))]
        (U,S,V) = np.linalg.svd(np.dot(vecVel.T, vecVel))
        sDir = V[-1]
        if sDir[0] > 0: # Should be approximately negative x
            # Flip around sDir
            sDir = -sDir
    else:
        sDir = np.asarray(sagittalDir)



    # Calculate the angle to the vertical. A positive angle means the vertical
    # vector is rotated to the direction of the segment vec by a positive
    # rotation about sDir. This means that vertDir, vec, sDir forms a
    # right-handed triple.

    return np.arcsin( np.dot(np.cross(vert, vec), sDir) )

    # Old stuff
    # Make sagittal plane horizontal
    sDir -= np.dot(vert, sDir)*vert
    sDir = sDir / np.linalg.norm(sDir)

    # The forward unit vector in the left-side sagittal plane
    fwd = np.cross(sDir, vert)

    #1/0
    # Calculating the angle
    return np.arctan2( np.dot(fwd, vec.T), np.dot(vert, vec.T) )


def inclination(acc, vertRef, horRef, g, gThreshold):
    """
    Calculates the inclination. The inclination is positive if the imu is
    rotated in the positive direction of the axis as given by self.d
    For the lower-limb this means that z points to the left, x points
    upwards, and y points forward. The angle phi is with respect to the
    x-direction and is positive when the g-vector is in the fourth
    (vertRef, -horRef) quadrant.
    """

    N = len(acc)
    tauK = [ k for k in range(N)
                    if np.isclose(np.linalg.norm(acc[k]), g, atol=gThreshold) ]
    yinc  =  np.array( [ np.arctan2(np.dot(acc[k], -horRef),
                                                    np.dot(acc[k], vertRef))
                                                                for k in tauK ] )
    return (tauK, yinc)

def angle_to_accref(q, acc, accref, gyro=None, magref=None):
    """
    Calculates the 3D angle between the g vector (taken as the average of the
    acceleration measurements rotated into a static frame) and the g vector
    taken as the acceleration at the reference measurement (standing still)

    Arguments
    q         ->  QuaternionArray with rotations of the IMU wrt the first data
                  sample
    acc       ->  (N,3) numpy array, acceleration measurements
    accref    ->  (N,) numpy array, acceleration at reference measurement
    gyro      ->  (N,3) gyro data. Used to determine a unit vector normal to the
                    main plane of movement (the sagittal plane). Default is
                    None, in which case an unsigned space angle is returned.
    magref    ->  (N,) reference magnetometer data at reference measurement. Used to determine sign of the unit vector
                    defining the plane of
    Returns
    phi       <-  (N,) numpy array, angle
    """


    # Rotate acceleration vectors
    acc_S = q.rotateVector(acc.T).T
    # Since the IMU returns to the same position, the average acceleration
    # must be one g pointing upwards.
    g = np.mean(acc_S, axis=0)
    gN = g / np.linalg.norm(g) # Normalize

    # Now rotate the g-vector according to the orientation q. This will given
    # The vertical direction upwards in the frame of the IMU.
    gb = q.rotateFrame(gN).T

    # accref gives the reference acceleration during standing still.
    # This is the reference vertical direction.
    gref = accref / np.linalg.norm(accref)

    if (gyro is None) or (magref is None):
        # Don't know how to determine the main plane of rotation/movement, and
        # so return unsigned angle
        return np.arccos(np.dot(gb, gref))
    else:
        # Determine the main plane of movement as the principle axis of the gyro
        # samples. Assume to be sagittal direction
        gcov = np.cov(gyro.T)
        eigenValues, eigenVectors = np.linalg.eig(gcov)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]

        sagittalDir = eigenVectors[:,0]
        sagittalDir *= np.sign(magref[2]*sagittalDir[2])

        # DEBUG
        #pyplot.figure()
        #pyplot.plot(acc_S)
        #pyplot.show()

        qarr = q

        #1/0
        # The z-component of the reference magnetometer is positive for left
        # side IMUs and negative for right-side IMUs. So the multiplication
        # with the sign function makes sure the
        # signal makes sure that the sagittal direction is positive local z
        # for the left side of the body and negative local z for the right side

        # Return a signed angle. Assume that a right-handed rotation about the
        # sagittal direction from current vertical to reference vertical. this
        # will correspond to a positive rotation from the reference to the
        # current orientation of the segment
        return np.arcsin( np.dot(np.cross(gb, gref), sagittalDir) )
