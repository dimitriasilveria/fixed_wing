import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm,expm
from icecream import ic
from utils import R3_so3, so3_R3

class FixedWing():
    def __init__(self):
        self.mb= 3.5
        self.g = 9.81
        self.h = 0.005
        self.dt = 0.01
        self.x = np.zeros(9)
        self.R = np.eye(3)
        Jx = 1.229
        Jy = 0.1702
        Jz = 0.8808
        Jxz = 0.9343
        self.J = np.array([[Jx, 0, -Jxz],
                    [0, Jy, 0, ],
                    [-Jxz, 0, Jz]
                    ])

        
    def _v_dot(self,v_b, w_b, f):
        return f/self.mb - np.cross(w_b,v_b)
    
    def _w_dot(self, w_b, tau):
        return np.linalg.inv(self.J)@(tau - np.cross(w_b, self.J@w_b))
    
    # def _R_dot(self, R, w):
    #     return R@R3_so3(w)
    
    def _R_update(self, w):
        self.R = self.R@expm(R3_so3(w)*self.h)
    
    def _v(self, v, R):
        return R@v

    def _f(self,t,x, inputs):
        w = x[0:3]
        v_b = x[3:6]
        f = inputs[0:3]
        tau = inputs[3:6]
        dv_dt = self._v_dot(v_b, w, f)
        dw_dt = self._w_dot(w, tau)
        dp_dt = self._v(v_b, self.R)
        return np.hstack((dw_dt, dv_dt, dp_dt))



        
    def _runge_kutta(self, x, inputs):
        N = int(self.dt/self.h)
        t = 0
        for i in range(N):
            t += self.h
            k1 = self.h*self._f(t ,x, inputs)
            k2 = self.h*self._f(t+0.5*self.h ,x + k1/2,inputs)
            k3 = self.h*self._f(t+0.5*self.h ,x + k2/2,inputs)
            k4 = self.h*self._f(t + self.h ,x + k3, inputs)
            w = x[0:3]
            self._R_update(w)
            x = x + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        return x
    
    def step(self, inputs):
        self.x = self._runge_kutta(self.x, inputs)
        return self.x, self.R

