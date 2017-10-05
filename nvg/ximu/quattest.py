""" Testing the quaternion package from Ling and Young """
import itertools
import numpy as np
import unittest
from nvg.maths import quaternions as quat
from nvg.algorithms import orientation

class TestQuat(unittest.TestCase):

    def setUp(self):
        self.vx = np.array([1.0, 0, 0])
        self.vy = np.array([0.0, 1, 0])

        self.vxm = np.asmatrix(self.vx).T
        self.vym = np.asmatrix(self.vy).T
    def test_euler(self):
        """
        Assuming that the quaternion represents the rotation of a frame from
        one orientation to another. Then the rotateVector method should take
        a vector represented in the original frame and assumed to be fixed
        in the rotating frame, and return its new orientation in the same original
        frame. In particular, if the rotation is a
        rotation of 90 degrees about the z-axis, then rotateVector operating on
        a unit vector in the x-direction should give a unit vector in the y-direction.
        """
        q = quat.Quaternion.fromEuler([0, 0, 90], 'xyz')
        R = q.toMatrix()
        vvx = q.rotateVector(self.vxm)
        vvx2 = np.dot(R,self.vx)
        self.assertEqual(vvx[1], 1.0)
        self.assertEqual(vvx[1], vvx2[0,1])

        # rotateFrame is the inverse of rotate vector. So when operating on a
        # unit y-vector it should return a unit x-vector.
        vvy = q.rotateFrame(self.vym)
        self.assertEqual(vvy[0], 1.0)

    def test_rotate_frame(self):
        """
        The quaternion can be viewed as representing the rotation of a frame B
        with respect to a frame A. The method rotateFrame should then, when
        operating on a vector represented in A give its representation in B.
        """
        eulers = np.array([[0,0,45],[0,0,90], [0, 0, 135], [0, 0, 180]])
        qa = quat.QuaternionArray([quat.Quaternion.fromEuler(ep, 'xyz') for ep in eulers])

        vs = np.column_stack((self.vx, self.vy, self.vx, self.vy))

        vvs = qa.rotateFrame(vs)
        vvsT = qa.rotateFrame(vs.T)

        print vvs
        print vvsT
        #self.assertTrue(any(vvs.T == vvsT))



    def test_gyro_integrator(self):
        """
        Define a constant positive angular velocity in the z-direction. integrating
        this should give a sequence of quaternions that when operating on a vector
        with rotateVector should rotate these in the positive direction according
        to the right hand rule with thumb pointing in the z-direction.

        Assume at the same time we are measuring a constant acceleration in the
        x-direction (locally). In the spatial frame, this should correspond to the
        directions given by rotateVector.
        """
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

        vx = np.repeat(self.vxm, n+1, axis=0)
        vxr = orientFilter.rotation.values.rotateVector(vx).T
        self.assertTrue(vxr[1,1] > 0)
        self.assertTrue(vxr[3,1] > 0)
        self.assertTrue(vxr[5,1] > 0)


        # Conclusion:
        # The orientation estimated by the orientfilter is the sequence of
        # quaternions that when used to rotate vectors in the body-frame gives
        # the orientation of these vectors in the spatial frame

if __name__ == '__main__':
    unittest.main()
