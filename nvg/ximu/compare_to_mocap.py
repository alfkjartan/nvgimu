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
import numpy.testing as npt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import itertools
from nvg.io import qualisys_tsv as qtsv
from nvg.ximu import ximudata as xdt

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
 

def compare_knee_RoM(xdb, mocapdata, subject, trial="N", resdir="/home/kjartan/Dropbox/project/nvg/resultat/compare-mocap", peakat=70):
    """
    Compares Range of motion of knee flexion for trial N (Normal condition).
    """
    startTime = 1*60 # s
    anTime = 180 # s

    # Load cycledata from imu data
    imudt = xdb.get_imu_data(subject, trial, "LA", startTime, anTime)
    firstPN = imudt[0][0,0]
    lastPN = imudt[0][-1,0]
    dt = 1.0/262.0 # Weird, but this is the actual sampling period
    cycledta_no_shift= xdb.get_cycle_data(subject, trial, 'LA', firstPN, lastPN)
    syncLA = xdb.get_PN_at_sync(subject, 'LA')
    cycledta = [[(t[0]-syncLA[0])*dt, (t[1]-syncLA[0])*dt] for t in cycledta_no_shift]
    firstCycleStart = cycledta[0][0]
    lastCycleEnd = cycledta[-1][1]

    md = mocapdata
    ankle = md.marker('ANKLE')
    knee = md.marker('KNEE')
    thigh = md.marker('THIGH')
    hip = md.marker('HIP')

    timeSinceSync = md.timeStamp - md.syncTime 

    frames2use = md.frameTimes[md.frameTimes>firstCycleStart-timeSinceSync.total_seconds()]
    frames2use = frames2use[frames2use<lastCycleEnd-timeSinceSync.total_seconds()]
    
    ankled = ankle.position(frames2use)
    kneed = knee.position(frames2use)
    hipd = hip.position(frames2use)
    thighd = thigh.position(frames2use)

    
    #jointangle = _three_point_angle( hipd, kneed, ankled, np.array([-1.0,0,0]) )
    #jointangle = _three_point_angle_projected( thighd, kneed, ankled, np.array([-1.0,0,0]) )
    jointangle = _four_point_angle_projected( hipd, thighd, ankled, kneed, np.array([-1.0,0,0]) )

    ft = frames2use + timeSinceSync.total_seconds()

    # Split and normalize to 100 datapoints. 
    x = np.linspace(0,100, 100)
    jointangleSplit = []
    for (cstart,cstop) in cycledta:
        (indStart,) = np.nonzero(ft < cstart)
        (indEnd,) = np.nonzero(ft > cstop)
        if len(indStart) == 0:
            indStart = [0]
        if len(indEnd) == 0:
            indEnd = [len(ft)-1]

        ja = jointangle[indStart[-1]:indEnd[0]]
        x0 = np.linspace(0,100, len(ja))
        f = interp1d(x0, ja, kind='linear')
        jointangleSplit.append(f(x))
                               
                               
    #accmagn = np.sqrt(np.sum(imudt[0][:,4:7]**2, axis=1))
    #imufr = imudt[0][:,0] - syncLA[0] 
                               
    angleBetweenSegments = xdb.get_angle_between_segments(subject, trial, ["LA", "LT"],
                                                          startTime=startTime, anTime=anTime, 
                                                          doPlots=False)

    
    # Normalize by finding positive peak, setting angle to zero at negative peak, determining time vector
    # with zero at peak, and index of peak

    kneeangle_md = []
    tvec = np.linspace(0,100,100)
    for ja in jointangleSplit:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        (negpeakind,) = np.nonzero(ja == np.min(ja))
        negpeakind = negpeakind[0]
        t = tvec - peakind
        angle = ja - ja[negpeakind]
        kneeangle_md.append( (angle, t, peakind) )

    kneeangle_imu = []
    for ja in angleBetweenSegments:
        (peakind,) = np.nonzero(ja == np.max(ja))
        peakind = peakind[0]
        (negpeakind,) = np.nonzero(ja == np.min(ja))
        negpeakind = negpeakind[0]
        t = tvec - peakind
        angle = ja - ja[negpeakind]
        kneeangle_imu.append( (angle, t, peakind) )


    plt.figure()
    for ja in kneeangle_md:
        plt.plot(ja[1], ja[0]*180/np.pi)
    plt.title("Knee angle from marker data") 
    plt.ylabel("Degrees")

    plt.figure()
    for ja in kneeangle_imu:
        plt.plot(ja[1], ja[0]*180/np.pi)
    plt.title("Knee angle from imu data") 
    plt.ylabel("Degrees")

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

    for (mda,imua) in itertools.izip(kneeangle_md, kneeangle_imu):
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
    plt.title('Knee flexion. Ensemble mean +/- 2 std  ' + subject + ' trial ' + trial )
    plt.ylabel('Degrees')
    plt.xticks([])
    plt.xlabel('One gait cycle')
    
    plt.savefig(resdir + '/' + subject + '_' + trial + '_knee-angle-mean-std.pdf')

            
    plt.figure()
    bland_altman_plot(mdflat.ravel(), imuflat.ravel())
    plt.title('Bland-Altman Plot')
  
    plt.show()

def compare_foot_clearance(xdb, mocapdata, subject, trial="N"):
    """
    Compares foot clearance per step for all steps during two minutes in middle of 
    trial N (Normal condition).
    """

    md = mocapdata
    anklepos = md.marker('ANKLE')

    timeSinceSync = md.timeStamp - md.syncTime 

    ankleposz = anklepos.position(md.frameTimes).transpose()[:,2]
    ft = md.frameTimes + timeSinceSync.total_seconds()

    # Load cycledata from imu data and check if ok wrt marker data
    cycledta = xdb.get_cycle_data(subject, trial, 'LA', 0, 800000000)
    syncLA = xdb.get_PN_at_sync(subject, 'LA')

    imudt = xdb.get_imu_data(subject, trial, "LA", 0, 500)
    accmagn = np.sqrt(np.sum(imudt[0][:,4:7]**2, axis=1))
    imufr = imudt[0][:,0] - syncLA[0] 
    

    plt.figure()
    plt.plot(ft, ankleposz)
    dt = 1.0/262.0
    for ind in cycledta:
        plt.plot([dt*float(ind[0]-syncLA[0]), dt*float(ind[0]-syncLA[0])], [-1, 1], 'm')
        plt.plot([dt*float(ind[1]-syncLA[0]), dt*float(ind[1]-syncLA[0])], [-1, 1], 'c')

    plt.plot(imufr*dt, accmagn, 'r')

    plt.show()

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
    v2 = p2-pcentral

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
            theta[i] = 2*np.pi - theta[i]

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

def main():
    resdir = "/home/kjartan/Dropbox/projekt/nvg/resultat/compare-mocap/2014-08-08"

    db = xdt.NVGData('/home/kjartan/Dropbox/Public/nvg201209.hdf5')

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

def check_PN_vs_time():
    dtapath = "/home/kjartan/Dropbox/projekt/nvg/data/test0910/6522/"
    dtapath = "/home/kjartan/nvg/2012-09-20/S7/"
    dtaset = "NVG_2012_S7_A_LT_00406";
    tk = xdt.TimeKeeper(dtapath + dtaset + "_DateTime.csv")
    
    t0 = tk.ttable[0][1]
    pns = np.array([row[0] for row in tk.ttable])
    times = np.array([(row[1]-t0).total_seconds() for row in tk.ttable])

    plt.figure()
    plt.plot(pns, times)
    plt.show()
    # OK! Straight line

    #Seconds per PN
    dt = (times[-1] - times[0]) / (pns[-1] - pns[0])
    print "dt = %0.4f s\n" % (dt,)
    print "samplefreq= %0.4f Hz\n" % (1.0/dt,)

    
    return pns,times
    
if __name__ == '__main__':
    
    print 'main'
    #unittest.main()
