""" Testing the quaternion package from Ling and Young """
import itertools
import numpy as np
import unittest
from imusim.maths import quaternions as quat
from imusim.algorithms import orientation

class TestQuat(unittest.TestCase):

    def setUp(self):
        self.vx = np.array([1.0, 0, 0])
        self.vy = np.array([0.0, 1, 0])

        self.vxm = np.asmatrix(self.vx).T
        self.vym = np.asmatrix(self.vy).T
    def test_euler(self):
        q = quat.Quaternion.fromEuler([0, 0, 90], 'xyz')
        R = q.toMatrix()
        vvx = q.rotateVector(self.vxm)
        vvx2 = np.dot(R,self.vx)
        self.assertEqual(vvx[1], 1.0)
        self.assertEqual(vvx[1], vvx2[0,1])

        vvy = q.rotateFrame(self.vym)
        self.assertEqual(vvy[0], 1.0)

    def test_rotate_frame(self):
        eulers = np.array([[0,0,45],[0,0,90], [0, 0, 135], [0, 0, 180]])
        qa = quat.QuaternionArray([quat.Quaternion.fromEuler(ep, 'xyz') for ep in eulers])

        vs = np.column_stack((self.vx, self.vy, self.vx, self.vy))

        vvs = qa.rotateFrame(vs)
        vvsT = qa.rotateFrame(vs.T)
        
        print vvs
        print vvsT
        #self.assertTrue(any(vvs.T == vvsT))
        
        

    def test_gyro_integrator(self):
        n = 100
        t = np.linspace(10.0/n,10, n)
        wm = np.pi/t[-1]/2.0
        w = np.repeat([[0.0, 0.0, wm]], n, axis=0)
        z = np.repeat(0.0, 3)

        initQ = quat.Quaternion()
        orientFilter = orientation.GyroIntegrator(0, initQ)
        
        for [tt, ww] in itertools.izip(t,w) :
            orientFilter(z, z, ww, tt)

        print orientFilter.rotation.latestValue.toMatrix()
        print type(orientFilter.rotation.values)
        
        # Conclusion:
        # The orientation estimated by the orientfilter is the sequence of 
        # quaternions that when used to rotate vectors in the body-frame gives
        # the orientation of these vectors in the spatial frame

if __name__ == '__main__':
    unittest.main()
        
