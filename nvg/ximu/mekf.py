
"""
Implementation of the multiplicative Kalman filter"""
#
# Translated from the matlab code by Fredrik Olsson
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

import numpy.testing as npt
import numpy as np

def measurement_update(x, P, acc, g0, Ra, mag, m0, Rm, type='quat'):
    """ Function that performs the measurement update in the multiplicative
        extended Kalman filter (MEKF) for computing orientation estimates using
        accelerometer and/or magnetometer measurements.

        Input:
        x       -   predicted states (t|t-1)
        P       -   predicted covariance (t|t-1)
        acc     -   accelerometer measurements
        g0      -   gravity vector in navigation frame
        Ra      -   accelerometer covariance
        mag     -   magnetometer measurements
        m0      -   magnetic north in navigation frame
        Rm      -   magnetometer covariance
        type    -   'quat' for quaternion states 'err' for quaternion error
                     states ('quat' is default)

        Output:
        x   -   filtered states (t|t)
        P   -   filtered covariance (t|t)

        """


        Rt = x.toMatrix()


        if type == 'err':
            #Multiplicative extended Kalman filter (MEKF) with quaternion error states
            Ca = -crossMat(Rt*g0);
            Cm = -crossMat(Rt*m0);

    if rank(Ra) == 3 && rank(Rm) == 3
        R = [Ra zeros(3);
             zeros(3) Rm];
        C = [Ca Cm];
        y = [acc; mag];
        yp = [Rt*g0; Rt*m0];
    elseif rank(Ra) == 3
        R = Ra;
        C = Ca;
        y = acc;
        yp = Rt*g0;
    elseif rank(Rm) == 3
        R = Rm;
        C = Cm;
        y = mag;
        yp = Rt*m0;
    else
        return;
    end

    % Compute error state and covariance
    C = C';
    eta = P*C'/(C*P*C'+R)*(y-yp); % Error state
    Pt = P - P*C'/(C*P*C'+R)*C*P; % Error state covariance

    % Relinearize (update quaternion state and covariance)
    if norm(eta) ~= 0
        deta = [cos(norm(eta)/2); eta/norm(eta)*sin(norm(eta)/2)];
    else
        deta = [1 0 0 0]';
    end
    x = quatmultiply(x',deta')';
    J = eye(3) - (1/2)*crossMat(eta);
    P = J*Pt*J';
else
    %% Extended Kalman filter (EKF) with quaternion states
    q0 = x(1); q1 = x(2); q2 = x(3); q3 = x(4);
    dQ1 = [ 4*q0 4*q1    0     0;
           -2*q3 2*q2 2*q1 -2*q0;
            2*q2 2*q3 2*q0  2*q1];
    dQ2 = [ 2*q3  2*q2 2*q1 2*q0;
            4*q0     0 4*q2    0;
           -2*q1 -2*q0 2*q3 2*q2];
    dQ3 = [-2*q2 2*q3 -2*q0 2*q1;
            2*q1 2*q0  2*q3 2*q2;
            4*q0    0     0 4*q3];

    Ca = g0(1)*dQ1 + g0(2)*dQ2 + g0(3)*dQ3;
    Cm = m0(1)*dQ1 + m0(2)*dQ2 + m0(3)*dQ3;
    if rank(Ra) == 3 && rank(Rm) == 3
        R = [Ra zeros(3);
             zeros(3) Rm];
        C = [Ca; Cm];
        yt = [acc - Rt*g0; mag - Rt*m0];
    elseif rank(Ra) == 3
        R = Ra;
        C = Ca;
        yt = acc - Rt*g0;
    elseif rank(Rm) == 3
        R = Rm;
        C = Cm;
        yt = mag - Rt*m0;
    else
        return;
    end

    S = C*P*C' + R;
    K = (P*C')/S;
    x = x + K*yt;
    P = (eye(size(K*C)) - K*C)*P;

end

def crossMat(x):
    return np.array([
                [ 0, -x[2], x[1] ],
                [ x[2],  0,  -x[0] ],
                [ -x[1], x[0], 0] ] )
