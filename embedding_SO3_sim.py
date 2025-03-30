import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
import math
from utils import R3_so3, so3_R3
from scipy.linalg import expm, logm
from icecream import ic
#the rotation is clockwise #################################
class Embedding():
    def __init__(self,r,phi_dot,k_phi,tactic,n_agents,dt):
        self.phi_dot = phi_dot
        self.r = r
        self.k_phi = k_phi
        self.tactic = tactic
        self.n = n_agents
        self.dt = dt
        self.Rot_des = np.zeros((3,3,self.n))
        self.Rot_act = np.zeros((3,3,self.n))
        if (self.tactic == 'circle') or (self.tactic == 'spiral'):
            self.scale = 0 #scale the distortion around the x axis
        else:
            self.scale = 0.01
        self.hover_height = 2*r
        self.count = 0
        for i in range(self.n):
            self.Rot_des[:,:,i] = np.eye(3)
            self.Rot_act[:,:,i] = np.eye(3)

        self.wd = np.zeros(self.n)
        self.T = 24
        self.t = np.zeros(self.n)
        self.timer = 1
        self.phi_des = np.zeros(self.n)
        self.phi_cur = np.zeros(self.n)
        self.phi_dot_actual = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.target_r = np.zeros((3, self.n))
        self.target_v = np.zeros((3, self.n))
        self.target_a = np.zeros((3, self.n))
       
    def targets(self, agent_r, agent_v):
        unit = np.zeros((self.n, 3))

        unit = np.zeros((self.n, 3))
        if self.n >1:
            n_diff = int(np.math.factorial(self.n) / (math.factorial(2) * math.factorial(self.n-2)))
        else:
            n_diff = 1
        phi_diff = np.zeros(n_diff)
        distances = np.zeros(n_diff)
        unit = np.zeros((self.n, 3))

        pos_circle = np.zeros((3, self.n))
        
        for i in range(self.n):
            if self.tactic == 'spiral':
                self.z[i] = self.hover_height*self.t[i]
            else:
                self.z[i] = self.hover_height

            # Circle position
            # pos = np.array([agent_r[0, i] - self.phi_dot*np.cos(phi_prev[i])*np.sin(phi_prev[i]), agent_r[1, i] - self.r*np.cos(phi_prev[i])**2, agent_r[2, i]-0.6])
            pos = np.array([agent_r[0, i] , agent_r[1, i] , agent_r[2, i]-self.z[i]])

            pos_rot = self.Rot_des[:,:,i].T@pos.T
            phi, _ = self.cart2pol(pos)


            pos_x = pos_rot[0]
            pos_y = pos_rot[1]
            #pos_x, pos_y, _ = pos_rot.parts[1:]  # Ignoring the scalar part
            self.phi_dot_actual[i] = (phi - self.phi_cur[i])/self.dt
            self.phi_cur[i] = phi
            pos_circle[0, i] = pos_x
            pos_circle[1, i] = pos_y
            unit[i, :] = [np.cos(phi), np.sin(phi), 0]

        if self.tactic == 'spiral':
            self.t += self.dt*np.ones(self.n)

            
        for i in range(self.n):
            if self.tactic == 'spiral':
                self.z[i] = self.hover_height*self.t[i]
            else:
                self.z[i] = self.hover_height
            if self.n > 1:
                phi_i = self.phi_cur[i]
                if i == 0:
                    phi_k = self.phi_cur[self.n-1] #ahead
                    phi_j = self.phi_cur[i+1] #behind
                elif i == self.n-1:
                    phi_k = self.phi_cur[i-1]
                    phi_j = self.phi_cur[0]
                else:
                    phi_k = self.phi_cur[i-1]
                    phi_j = self.phi_cur[i+1]


                wd = self.phi_dot_desired(phi_i, phi_j, phi_k, self.phi_dot, self.k_phi,i)
            else:
                wd = self.phi_dot
                phi_i = self.phi_cur[0]
            #wd = self.phi_dot
            
            #first evolve the agent in phase
            v_d_hat_z = np.array([0, 0, wd])
            x = self.r * np.cos(phi_i)
            y = self.r * np.sin(phi_i)
            Rot_z = expm(R3_so3(v_d_hat_z)*self.dt)
            pos_d_hat = np.array([x, y, 0])
            pos_d_hat = Rot_z@pos_d_hat
            phi_d, _, = self.cart2pol(pos_d_hat)

            phi_dot_x = self.calc_wx(phi_d)#*(phi_d-self.phi_des[i])
            phi_dot_y = self.calc_wy(phi_d) #phi_i-phi_prev[i]*
            v_d_hat_x_y = np.array([phi_dot_x, phi_dot_y, 0])
            self.Rot_des[:,:,i] = expm(R3_so3(v_d_hat_x_y))
     

            pos_d = self.Rot_des[:,:,i]@pos_d_hat
            # if i == 1 and not self.pass_ref[i]:
            pos_d += np.array([0,0,self.z[i]])

            # vel_d = (pos_d - np.array([x, y, self.z[i]]))/self.dt

            # accel_d = (vel_d - agent_v[:,i])/self.dt
            vel_d = (pos_d - self.target_r[:,i])/self.dt

            accel_d = (vel_d - self.target_v[:,i])/self.dt

            self.target_r[:,i] = pos_d
            self.target_v[:,i] = vel_d
            self.target_a[:,i] = accel_d

            # if self.tactic == 'circle':
            #     target_r[2,i] = 0.5
            unit[i, :] = [np.cos(phi_i), np.sin(phi_i), 0]

            self.phi_des[i] = phi_d
        # with warnings.catch_warnings(record=True) as w:
        #     warnings.simplefilter("always")  # Ensure all warnings are captured
        k = 0
        if self.n == 2:
            distances[0] = np.linalg.norm(self.target_r[:, 0] - self.target_r[:, 1])
            phi_diff[0] = np.abs(self.phi_cur[0] - self.phi_cur[1])
        for i in range(self.n):
            for j in range(i+1, self.n):
                distances[k] = np.linalg.norm(self.target_r[:, i] - self.target_r[:, j])
                phi_diff[k] = np.arccos(np.dot(unit[i,:],unit[j,:]))
                k += 1
        
        
        return  self.phi_cur,self.target_r, self.target_v, self.target_a, phi_diff, distances
    

    def calc_wx(self,phi):
        # return self.scale*(np.sin(phi)*np.cos(phi)-np.sin(phi)**3)
        return self.scale*np.cos(phi)*np.sin(phi)
    
    def calc_wy(self,phi):
        return 0*self.scale*np.sin(phi)*np.cos(phi)**2

    def phi_dot_desired(self,phi_i, phi_j, phi_k, phi_dot_des, k,i):

        R_i = R.from_euler('z', phi_i, degrees=False).as_matrix()
        R_j = R.from_euler('z', phi_j, degrees=False).as_matrix()
        R_k = R.from_euler('z', phi_k, degrees=False).as_matrix()
        R_ji = R_j.T@R_i
        R_ki = R_k.T@R_i

        w_diff_ji = so3_R3(logm(R_ji.T))[2]
        w_diff_ki = so3_R3(logm(R_ki.T))[2]
        if w_diff_ji == 0:
            w_diff_ji = 0.00001
        if w_diff_ki == 0:
            w_diff_ki = 0.00001

        phi_dot_des = self.phi_dot +  k*(1/(w_diff_ji.real) + 1/(w_diff_ki.real)) # 0.1*(w_neg.real + w_pos.real) #+ np.clip(-0.5/(w_diff_ij.real) + 0.5/(w_diff_ki.real),-0.5,0.5)


        return np.clip(phi_dot_des,0.1,0.5)


    def cart2pol(self,pos_rot):
        pos_x = pos_rot[0]
        pos_y = pos_rot[1]
        #pos_x, pos_y, _ = pos_rot.parts[1:]
        phi_raw = np.arctan2(pos_y, pos_x)
        phi = np.mod(phi_raw, 2*np.pi)
        r = np.linalg.norm([pos_x, pos_y])
        return phi, r


    # Quaternion multiplication function (can be skipped if using numpy.quaternion)
    def quat_mult(self,q1, q2):
        return np.quaternion(
            q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
        )
    
    
        
