import numpy as np
from utils import R3_so3, dX_to_dXi, so3_R3, SE3_se3_back
from scipy.linalg import expm, logm
from numpy.linalg import matrix_rank
from scipy.linalg import block_diag
from fixed_wing import FixedWing
import control as ctrl
import pandas as pd
# from scipy.signal import place_poles
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import os
from icecream import ic
import csv
from scipy.sparse import csc_matrix
from cvxpy import psd_wrap
from scipy.optimize import minimize, Bounds
import cvxpy as cp
from scipy.linalg import block_diag

#Initiating constants##############################################
class MPC_fixed_wing():
    def __init__(self,t_max, t_min, r, T_p, id, figures_path, noise = False, save = False):
        #fixed wing model
        self.id = id
        self.noise = noise
        self.save = save
        self.figures_path = figures_path
        self.fixed_wing = FixedWing("/home/dimitria/fixed_wing/pyfly/pyfly/pyfly_config.json", "/home/dimitria/fixed_wing/pyfly/pyfly/x8_param.mat")
        self.n = 12
        self.m = 6
        self.qsi = np.zeros((self.n,1)) 
        self.mb = self.fixed_wing.mb
        self.g = self.fixed_wing.g
        self.J = self.fixed_wing.J
        self.ga = np.array([0, 0, -self.g])

        self.I3 = np.array([0, 0, 1])
        
        self.r = r
        self.T_p = T_p
        # self.r = 100
        # self.T_p = 25
        self.t_max =t_max
        self.t_min = t_min
        
        self.T = self.fixed_wing.dt
        self.N = int((self.t_max-self.t_min)/self.T) #number of points
        ic(self.N)
        #reference trajectories###################################################

        self.wr_r = np.zeros((3,self.N))  #reference angular velocity
        self.d_wr_r = np.zeros((3,self.N))  #reference angular acceleration
        self.Car = np.zeros((3,3,self.N))  #reference attitude that converts from body fame to inertial
        self.angels = np.zeros((3,self.N))
        self.des_angles = np.zeros((3,self.N))
        #self.Car[:,:,0] = np.eye(3) #eul2rotm([pi*0.5 0 0])*Ca_r(:,:,1) #
        self.z_w = np.array([0,0,1])
        self.f_r= np.zeros((3,self.N))  #reference force
        self.tau_r = np.zeros((3,self.N))  #reference torque
        # self.f_A_r = np.zeros((3,self.N))  #reference aerodynamics force
        #controller and state-space matrices ###################################
        self.A = np.zeros((self.n,self.n,self.N))
        self.B = np.zeros((self.n,self.m,self.N)) #check this

        self.t = np.linspace(self.t_min,self.t_max,self.N) #time
        self.ra_r = np.zeros((3,self.N))  #reference position
        self.va_r = np.zeros((3,self.N))  #reference linear velocity
        self.va_r_dot = np.zeros((3,self.N))  #reference linear acceleration
        self.va_r_dot_body = np.zeros((3,self.N))
        self.va_r_body = np.zeros((3,self.N))
        # p = np.array([-5,-2.5, -1.1,-2.3,-0.5,-1.5,-2.2,-3.1,-2])
        # p = np.array([-5,-2.5, -10.1,-2.3,-0.5,-1.5,-2.2,-3.1,-2,-6.5,-3.4,-8])
        self.Nh = 5
        self.Q_v = 10**(3)*np.eye(3)
        self.Q_r = 1e7*np.eye(3)
        self.Q_phi =1e7*np.eye(3)
        self.Q_aug = 1e1*np.eye(3)
        self.Q_f = 1e2
        self.Q_w = 1e4
        # self.Q_v = 10**(5)*np.eye(3)
        # self.Q_r = 1e9*np.eye(3)
        # self.Q_phi =1e2*np.eye(3)
        # self.Q_aug = 1e1*np.eye(3)
        # self.Q_f = 1e5
        # self.Q_w = 10**(5)
        self.Q_i = np.eye(self.n)
        if self.n == 12:
            self.Q_i[0:12,0:12] = block_diag(self.Q_phi, self.Q_v, self.Q_r, self.Q_aug)
        else:
            self.Q_i[0:9,0:9] = block_diag(self.Q_phi, self.Q_v, self.Q_r)
        # Q_f = 1/(1)
        # Q_w = 1/(0.5)
        self.c1 = 0.2


        #actual trajectory and errors ###########################################
        self.Cab = np.zeros((3,3,self.N))
        self.Cab[:,:,0] = np.eye(3) #eul2rotm([pi*0.5 0 0])*Ca_r(:,:,1) #
        self.f = np.zeros((3,self.N))  #actual force
        self.f_r_body = np.zeros((3,self.N))  #actual force in body frame
        self.tau = np.zeros((3,self.N))  #actual torque
        # self.f_A = np.zeros((3,self.N))  #actual aerodynamics force
        self.wb_b_cont = np.zeros((3,self.N)) #angular velocity input
        self.dwb_b_cont = np.zeros((3,self.N)) #angular acceleration input
        self.d_Xi = np.zeros((self.n,self.N))  #error
        self.dU = np.zeros((self.m,self.N))
        self.dC = np.zeros((3,3,self.N))
        # self.d_v = np.zeros(3)
        # self.d_r = np.zeros(3)
         
        self.abs_phi = np.zeros((1,self.N))
        self.abs_r = np.zeros((1,self.N))
        self.abs_v = np.zeros((1,self.N))
        self.abs_f = np.zeros((1,self.N))
        self.abs_f_a = np.zeros((1,self.N))
        self.abs_w = np.zeros((1,self.N))

        self.X = np.zeros((9,self.N)) 
        
        self.f_min = None
        self.f_max = None
        self.tau_min = None
        self.tau_max = None
        # self.d_Xi[9:12,0] = self.c1*self.d_Xi[6:9,0] + self.d_Xi[3:6,0] 

            
    def references(self,i, ra_r, va_r, va_r_dot):
        self.ra_r[:,i] = ra_r
        self.va_r[:,i] = va_r
        self.va_r_dot[:,i] = va_r_dot
        # if i >=1:
        #     self.va_r[:,i] = (self.ra_r[:,i]-self.ra_r[:,i-1])/self.T
        #     self.va_r_dot[:,i] = (self.va_r[:,i]-self.va_r[:,i-1])/self.T
        # else:
        #     self.va_r[:,i] = va_r
        #     self.va_r_dot[:,i] = va_r_dot
        norm_v = np.linalg.norm(self.va_r[:,i])
        v_norm = self.va_r[:,i]/norm_v

        # v_xy_norm = np.linalg.norm(self.va_r[0:2,i])
        # a_perpendicular = self.va_r_dot[:,i] - np.dot(self.va_r_dot[:,i],v_norm)*v_norm
        # phi = np.arctan2(a_perpendicular[1],self.g+a_perpendicular[2])
        # phi = np.arctan2(np.linalg.norm(a_perpendicular),self.g)


        
        #pitch (theta)
        # theta = np.arctan2(self.va_r_dot[2,i],v_xy_norm)
        if np.linalg.norm(self.va_r[:,i]) == 0:
            theta = 0
            psi = 0
        else:
            psi = np.arctan2(v_norm[1],v_norm[0])
            theta = -np.arcsin(self.va_r[2,i]/norm_v)
        phi = np.arctan2(norm_v**2*np.cos(theta),self.r*self.g)


        if i == 2:
            ic(np.rad2deg(phi))
            ic("minimum lift: ", self.mb*self.g/np.cos(phi))

            # Rotation matrices using Z-Y-X intrinsic rotations
        Rz = np.array([
            [np.cos(psi),-np.sin(psi), 0],
            [np.sin(psi), np.cos(psi),  0],
            [0,           0,            1]
        ])

        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0,             1, 0],
            [-np.sin(theta),0, np.cos(theta)]
        ])

        Rx = np.array([
            [1, 0,          0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi),  np.cos(phi)]
        ])


        self.Car[:,:,i+1] =  (Rz @ Ry @ Rx) #@Rx_180
        self.des_angles[:,i] = np.array([phi,theta,psi])

        if i == 0:
            self.Car[:,:,i] = self.Car[:,:,i+1]@R.from_euler('xyz', [0, 0, 0.01]).as_matrix()

        # self.Car[:,:,i] = np.hstack((x_B.reshape(3,1), y_B.reshape(3,1), z_B.reshape(3,1)))#@R.from_euler('xyz', [np.pi/100, 0, 0]).as_matrix()
        if i > 1 :

            mat= self.Car[:,:,i].T@self.Car[:,:,i+1]
            self.wr_r[:,i] = so3_R3(logm(mat))/(1*self.T)
            
            
            # self.wr_r[:,i] = np.clip(self.wr_r[:,i],-np.pi,np.pi)
            v_body = self.Car[:,:,i].T@self.va_r[:,i]
            a_body  = self.Car[:,:,i].T@self.va_r_dot[:,i]
            # g_body = self.Car[:,:,i].T@self.ga
            # self.wr_r[1,i] = -(a_body[2]+g_body[2])/norm_v
            # self.wr_r[2,i] = g_body[1]/norm_v
            self.va_r_dot_body[:,i] = a_body
            self.va_r_body[:,i] = v_body
            # wr_r_inertial = self.Calr[:,:,i]@self.wr_r[:,i]
            # self.f_r[:,i] = self.mb*(self.va_r_dot[:,i] - self.g*self.z_w  + np.cross(wr_r_inertial,self.va_r[:,i]))
            self.f_r[:,i] = self.mb*(a_body + np.cross(self.wr_r[:,i],self.va_r_body[:,i])) 
            if i > 2 :
                self.d_wr_r[:,i] = (self.wr_r[:,i] - self.wr_r[:,i-1])/(1*self.T)
            self.tau_r[:,i] = self.J@self.d_wr_r[:,i] + np.cross(self.wr_r[:,i], self.J@self.wr_r[:,i])

        else:
            a_body  = self.Car[:,:,i].T@self.va_r_dot[:,i]
            self.va_r_dot_body[:,i] = a_body
            self.va_r_body[:,i] = self.Car[:,:,i].T@self.va_r[:,i]
            self.f_r[:,i] = self.Car[:,:,i].T@(self.mb*self.va_r_dot[:,i] ) 
            # if np.linalg.det(self.Car[:,:,i]) != 0 and np.linalg.det(self.Car[:,:,i+1]) != 0:
                
            # else:
            #     self.wr_r[:,i] = np.zeros(3)


    def calc_A_and_B(self,i):
        self.A[:,:,i]   = np.zeros((self.n,self.n))
        self.B[:,:,i]   = np.zeros((self.n,self.m))
        self.A[3:6,0:3,i]   = R3_so3(self.wr_r[:,i])@R3_so3(self.va_r[:,i]) #- R3_so3(self.f_r[:,i])/self.mb
        self.A[3:6,3:6,i]   = R3_so3(self.wr_r[:,i])- R3_so3(self.wr_r[:,i])@self.Car[:,:,i]
        self.A[6:9,3:6,i]   = np.eye(3) 
        self.A[6:9,6:9,i]   = -R3_so3(self.wr_r[:,i]) 

        # self.B[5,0,i] = 1/self.mb 
        self.B[3:6, 0:3,i] = np.eye(3)/self.mb
        self.B[0:3, 3:6,i]  = np.eye(3)  
        self.B[3:6, 3:6,i]  = R3_so3(self.va_r[:,i])
        if self.n == 12:
            self.A[9:12,3:6,i]  = np.eye(3) 
            self.A[9:12,6:9,i]  = self.c1*np.eye(3) 
        #self.B[7, 0, i] = 1
        self.A[:,:,i] = np.eye(self.n) + self.T * self.A[:,:,i] 
        self.B[:,:,i] = self.T*self.B[:,:,i]#+ 1e-5 * np.random.randn(*self.B[:,:,i].shape)


    def get_MPC_matrices(self,i):

        Q_hat = np.zeros((self.n*self.Nh,self.n*self.Nh))
        R_hat = np.zeros((self.m*self.Nh,self.m*self.Nh))
        Aqp = np.zeros((self.n*self.Nh,self.n))
        Bqp = np.zeros((self.n*self.Nh,self.m*self.Nh))

        for j in range(0,self.Nh-1):
            Aqp[j*self.n:(j+1)*self.n,:] = np.linalg.matrix_power(self.A[:,:,i+j],(j+1))
            Q_hat[self.n*j:self.n*j+self.n,self.n*j:self.n*j+self.n] =  self.Q_i
            R_hat[self.m*j:self.m*j+self.m,self.m*j:self.m*j+self.m] = np.diag([self.Q_f,self.Q_f,self.Q_f,self.Q_w,self.Q_w,self.Q_w])

        for j in range(0,self.Nh-1):
            a = np.vstack([np.zeros(((j+1)*self.n,self.n)), np.eye(self.n), Aqp[:-(j+1)*self.n,:]])
            Bqp[:,self.m*j:self.m*j+self.m] = a[self.n:,:]@self.B[:,:,i+j]

        Q_hat[-self.n:,-self.n:] = self.Q_i
        return Q_hat, R_hat, Aqp, Bqp
    
    def mpc_outputs(self,i):
        Q_hat, R_hat, Aqp, Bqp = self.get_MPC_matrices(i)
        H =  Bqp.T @ Q_hat @ Bqp + R_hat
        H = (H + H.T) / 2  # Ensure symmetry

        # Convert to sparse matrices
        Aqp_sparse = csc_matrix(Aqp)
        H_sparse = csc_matrix(H)
        # Compute G

        G = 2 * Bqp.T @ Q_hat @ Aqp_sparse @ self.d_Xi[:,i]
        # Define the quadratic objective function: 0.5 * U^T H U + G^T U
        # Define bounds for each control input
        # U_min = np.array([-f_max, -f_max, -f_max, -np.pi, -np.pi, -np.pi])
        # U_min = np.tile(U_min, self.Nh)
        # U_max = np.array([f_max, f_max, f_max, np.pi, np.pi, np.pi])
        # U_max = np.tile(U_max, self.Nh)
            # bounds = [(U_min[i], U_max[i]) for i in range(len(U_min))]
        f_max = np.inf
        u_min_single = np.array([-f_max, -f_max, -f_max, -np.pi, -np.pi, -np.pi])
        u_max_single = np.array([f_max, f_max, f_max, np.pi, np.pi, np.pi])
        U_min = np.tile(u_min_single, self.Nh)
        U_max = np.tile(u_max_single, self.Nh)
        bounds = Bounds(U_min, U_max)
        def grad(U):
            return H @ U + G
        # Define the quadratic objective function: 0.5 * U^T H U + G^T U
        n = H.shape[0]
        U = cp.Variable(n)
        # constraints = [
        #     U >= U_min,
        #     U <= U_max

        # ]
        # constraints = [
        # {'type': 'ineq', 'fun': lambda U: U - U_min},  # U >= 0
        # {'type': 'ineq', 'fun': lambda U: U_max - U},  # U <= 10
        #     ]

        def objective(U):
            return 0.5 * U.T @ H_sparse @ U + G.T @ U
        # H += 1e-6 * np.eye(H.shape[0])
        # objective = cp.Minimize(0.5 * cp.quad_form(U, psd_wrap(H)) + G.T @ U)
        # prob = cp.Problem(objective, constraints)
        U0 = np.ones(H.shape[0])
        def hess(U):
            return np.zeros((len(U), len(U)))
        # res = minimize(objective, U0,hess=hess, method='trust-constr', constraints=constraints,
        #        options={'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 200, 'disp': False})
        # res = minimize(
        #     fun=objective,
        #     x0=U0,
        #     jac=grad,
        #     method='trust-constr',
        #     bounds=bounds,
        #     options={'gtol': 1e-4, 'xtol': 1e-4, 'maxiter': 200, 'disp': False}
        # )
        res =minimize(objective, U0, method='SLSQP', bounds=bounds, options={'disp': False, 'maxiter': 200, 'ftol': 1e-4})
        U_h = res.x
        # prob.solve(solver=cp.ECOS, abstol=1e-5, reltol=1e-5, max_iters=3000, verbose=False)
        # U_h = U.value

        # Extract first 4 elements for dU
        dU = U_h[:self.m]
        # ic(dU)

        # dU = U0[:self.m]

        return dU
    
    def set_initial_conditions(self, i):

        self.X[3:6,i] = self.va_r[:,i] 
        self.X[6:9,i] = self.ra_r[:,i] 
        self.fixed_wing.x[0:3] = self.wr_r[:,i]
        self.fixed_wing.x[6:9] = self.ra_r[:,i]

        # va_body = self.Car[:,:,i].T@self.va_r[:,i]

        self.fixed_wing.x[3:6] = self.Car[:,:,i].T@self.va_r[:,i]
        self.fixed_wing.R = self.Car[:,:,i]
        self.Cab[:,:,i] = self.fixed_wing.R
        self.dC[:,:,i] = self.Cab[:,:,i].T@self.Car[:,:,i]
        self.d_v = self.Cab[:,:,i].T@(self.va_r[:,i] - self.Car[:,:,i]@self.X[3:6,i]) 
        self.d_r = self.Cab[:,:,i].T@(self.ra_r[:,i] - self.X[6:9,i]) 
        self.d_Xi[0:9,i] = dX_to_dXi(self.dC[:,:,i],self.d_v,self.d_r)
        if self.n == 12: 
            self.d_Xi[9:12,i] = self.c1*self.d_Xi[6:9,i] + self.d_Xi[3:6,i] 
        self.wb_b_cont[:,i] = self.dC[:,:,i]@self.wr_r[:,i]

    def control(self, i, ra_r, va_r, va_r_dot):

        # self.references(i,ra_r, va_r, va_r_dot)
        
        # self.calc_K_pid(i)
        # self.fixed_wing.x[3:6] = self.Car[:,:,i].T@self.va_r[:,i]
        self.dC[:,:,i] = self.Cab[:,:,i].T@self.Car[:,:,i]
        # self.dC[:,:,i+1] = self.Cab[:,:,i+1].T@self.Car[:,:,i+1]
        self.d_v = self.Cab[:,:,i].T@(self.va_r[:,i] - self.X[3:6,i]) 
        self.d_r = self.Cab[:,:,i].T@(self.ra_r[:,i] - self.X[6:9,i]) 
        self.d_Xi[0:9,i] = dX_to_dXi(self.dC[:,:,i],self.d_v,self.d_r)
        if self.n == 12: 
            self.d_Xi[9:12,i] = self.c1*self.d_Xi[6:9,i] + self.d_Xi[3:6,i] 

        self.dU[:,i] = self.mpc_outputs(i)
        self.abs_phi[0,i] =np.linalg.norm(self.d_Xi[0:3,i]) 
        self.abs_r[0,i] =  np.linalg.norm(self.X[6:9,i]-self.ra_r[:,i]) 
        self.abs_v[0,i] =  np.linalg.norm(self.X[3:6,i] - self.va_r[:,i]) 
        self.abs_f[0,i] = np.linalg.norm(self.dU[0:3,i]) 
        self.abs_w[0,i] =np.linalg.norm(self.dU[3:6,i]) 
        #state constraints
        # self.f_min, self.tau_min = self.fixed_wing._forces_min(self.Car[:,:,i])
        # self.f_max, self.tau_max = self.fixed_wing._forces_max(self.Car[:,:,i])
        self.f[:,i] = self.dC[:,:,i]@self.f_r[:,i] - self.dU[0:3,i]
        # self.f[:,i] = self.f_r[:,i] - self.dU[0:3,i]
        self.f_r_body[:,i] = self.dC[:,:,i]@self.f_r[:,i]
        
        self.wb_b_cont[:,i] = self.dC[:,:,i]@self.wr_r[:,i] - self.dU[3:6,i]
        self.wb_b_cont[:,i] = np.clip(self.wb_b_cont[:,i],-np.pi,np.pi)
        if self.noise:
            self.f[:,i] = self.f[:,i]*np.random.normal(1,0.1,3)
            self.wb_b_cont[:,i] = self.wb_b_cont[:,i]*np.random.normal(1,0.1,3)
        if i > 2:
            self.dwb_b_cont[:,i] = (self.wb_b_cont[:,i] - self.wb_b_cont[:,i-1])/self.T
        self.tau[:,i] = self.J@self.dwb_b_cont[:,i] + np.cross(self.wb_b_cont[:,i], self.J@self.wb_b_cont[:,i])
        # self.tau[:,i] = np.clip(self.tau[:,i],self.tau_min,self.tau_max)
        #self.wb_b_cont[:,i] = np.clip(self.wb_b_cont[:,i],-np.pi/3,np.pi/3)
        # ic(i,'debug')
        x, Rot = self.fixed_wing.step(np.hstack((self.f[:,i],self.tau[:,i]))) 

        va_body = self.Car[:,:,i+1].T@self.va_r[:,i+1]
        # Save va_body to a CSV file

        #updating the states
        self.Cab[:,:,i+1] = Rot
        phi, theta, psi = R.from_matrix(self.Cab[:,:,i]).as_euler('xyz', degrees=False)
        self.angels[:,i+1] = np.array([phi,theta,psi])  #
        
        
        self.X[0:3,i+1] = x[0:3]
        self.X[6:9,i+1] = x[6:9]
        self.X[3:6,i+1] = self.Cab[:,:,i+1]@x[3:6]

        #printing the next desired values           


        ic(i)

    def plot_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.X[6,:-self.Nh],self.X[7,:-self.Nh],self.X[8,:-self.Nh],label = "real traj.",color = 'black')
        ax.plot(self.X[6,:-self.Nh],self.X[7,:-self.Nh],2*np.ones_like(self.X[7,:-self.Nh]),label = "XY proj.", color = 'red')
        ax.plot((-self.r - 2)*np.ones_like(self.X[7,:-self.Nh]), self.X[7,:-self.Nh],self.X[8,:-self.Nh],label = "YZ proj.",color = 'red')

        ax.plot(self.ra_r[0,:-self.Nh],self.ra_r[1,:-self.Nh],self.ra_r[2,:-self.Nh],label = "ref. traj.",linestyle='dashed', color = "blue")
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title("Agent " + self.id)
        # ax.set_zlim(-1,0.1)
        # ax.set_ylim(0, 5)
        # ax.set_xlim(190, 200)
        # ax.set_zlim(-1, 0.1)
        # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
        # ax.set_xlim(190,200)
        ax.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/3D_plot_{self.id}_fixed_wing.png")
        # plt.show()
            #would f_T be the same as f_T_r depending on what attitude I use?
    def plot_erros(self):
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.t[:-self.Nh],np.rad2deg(self.abs_phi[0,:-self.Nh]) , color = "blue")
        plt.ylabel(r"$\|\delta\theta\ \mathrm{(degrees)}\|$")

        plt.title("Agent " + self.id)
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(self.t[:-self.Nh],self.abs_v[0,:-self.Nh], color = "blue")
        plt.ylabel("$||\delta v (m/s)||$")
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(self.t[:-self.Nh],self.abs_r[0,:-self.Nh], color = "blue")
        plt.ylabel("$||\delta p (m)||$")
        plt.grid()
        # plt.plot(self.t[:-self.Nh],self.abs_w[0,:-self.Nh],label = "w error")
        # plt.plot(self.t[:-self.Nh],self.abs_f[0,:-self.Nh],label = "f error")
        plt.xlabel('t (s)')
        
        
        # plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/error_plot_{self.id}.png")
        # plt.show()
    def plot_force_omega(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t[:-self.Nh],self.f_r_body[0,:-self.Nh],label = "f_r x")
        plt.plot(self.t[:-self.Nh],self.f[0,:-self.Nh],label = "f x", linestyle='dashed')
        plt.plot(self.t[:-self.Nh],self.f_r_body[1,:-self.Nh],label = "f_r y")
        plt.plot(self.t[:-self.Nh],self.f[1,:-self.Nh],label = "f y", linestyle='dashed')
        plt.plot(self.t[:-self.Nh],self.f_r_body[2,:-self.Nh],label = "f_r z")
        plt.plot(self.t[:-self.Nh],self.f[2,:-self.Nh],label = "f z", linestyle='dashed')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.t[:-self.Nh],self.wb_b_cont[0,:-self.Nh],label = "omega x")
        plt.plot(self.t[:-self.Nh],self.wb_b_cont[1,:-self.Nh],label = "omega y")
        plt.plot(self.t[:-self.Nh],self.wb_b_cont[2,:-self.Nh],label = "omega z")
        plt.legend()
        plt.title("Agent " + self.id)
        if self.save:
            plt.savefig(f"{self.figures_path}/force_omega_plot_{self.id}.png")
        # plt.show()

    def plot_acceleration(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[:-self.Nh],self.va_r_dot_body[0,:-self.Nh],label = "a_x")
        plt.plot(self.t[:-self.Nh],self.va_r_dot_body[1,:-self.Nh],label = "a_y")
        plt.plot(self.t[:-self.Nh],self.va_r_dot_body[2,:-self.Nh],label = "a_z")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.t[:-self.Nh],self.va_r[0,:-self.Nh],label = "v_x")
        plt.plot(self.t[:-self.Nh],self.va_r[1,:-self.Nh],label = "v_y")
        plt.plot(self.t[:-self.Nh],self.va_r[2,:-self.Nh],label = "v_z")
        plt.legend()
        plt.title("Agent " + self.id)
        if self.save:
            plt.savefig(f"{self.figures_path}/acceleration_plot_{self.id}.png")
        
        # plt.show()
    def plot_controls(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[:-self.Nh],self.X[0,:-self.Nh],label = "w_actual_x")
        plt.plot(self.t[:-self.Nh],self.X[1,:-self.Nh],label = "w_actual_y")
        plt.plot(self.t[:-self.Nh],self.X[2,:-self.Nh],label = "w_actual_z")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.t[:-self.Nh],self.wr_r[0,:-self.Nh],label = "omega_p")
        plt.plot(self.t[:-self.Nh],self.wr_r[1,:-self.Nh],label = "omega_q")
        plt.plot(self.t[:-self.Nh],self.wr_r[2,:-self.Nh],label = "omega_r")
        plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/controls_plot_{self.id}.png")
        # plt.show()

    def plot_references(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[:-self.Nh],self.f[0,:-self.Nh],label = "f_x")
        plt.plot(self.t[:-self.Nh],self.f[1,:-self.Nh],label = "f_y")
        plt.plot(self.t[:-self.Nh],self.f[2,:-self.Nh],label = "f_z")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.t[:-self.Nh],self.wr_r[0,:-self.Nh],label = "omega_p")
        plt.plot(self.t[:-self.Nh],self.wr_r[1,:-self.Nh],label = "omega_q")
        plt.plot(self.t[:-self.Nh],self.wr_r[2,:-self.Nh],label = "omega_r")
        plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/references_plot_{self.id}.png")

    def plot_velocity(self):
        fig = plt.figure()
        plt.subplot(3 ,1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[1:-self.Nh],self.X[3,1:-self.Nh],label = "u")
        plt.plot(self.t[1:-self.Nh],self.va_r[0,1:-self.Nh],label = "u_r")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.t[1:-self.Nh],self.X[4,1:-self.Nh],label = "v")
        plt.plot(self.t[1:-self.Nh],self.va_r[1,1:-self.Nh],label = "v_r")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.t[1:-self.Nh],self.X[5,1:-self.Nh],label = "w")
        plt.plot(self.t[1:-self.Nh],self.va_r[2,1:-self.Nh],label = "w_r")
        plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/velocity_plot_{self.id}.png")


    def plot_velocity_body(self):
        fig = plt.figure()
        plt.subplot(3 ,1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[1:-self.Nh],self.va_r_body[0,1:-self.Nh],label = "u_r")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.t[1:-self.Nh],self.va_r_body[1,1:-self.Nh],label = "v_r")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.t[1:-self.Nh],self.va_r_body[2,1:-self.Nh],label = "w_r")
        plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/velocity_body_plot_{self.id}.png")
        # plt.show()
    def plot_angles(self):
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[:-self.Nh],np.rad2deg(self.angels[0,:-self.Nh]),label = "phi")
        plt.plot(self.t[:-self.Nh],np.rad2deg(self.des_angles[0,:-self.Nh]),label = "phi_r")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.t[:-self.Nh],np.rad2deg(self.angels[1,:-self.Nh]),label = "theta")
        plt.plot(self.t[:-self.Nh],np.rad2deg(self.des_angles[1,:-self.Nh]),label = "theta_r")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.t[:-self.Nh],np.rad2deg(self.angels[2,:-self.Nh]),label = "psi")
        plt.plot(self.t[:-self.Nh],np.rad2deg(self.des_angles[2,:-self.Nh]),label = "psi_r")
        plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/angles_plot_{self.id}.png")

    def plot_positions(self):
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[:-self.Nh],self.X[6,:-self.Nh],label = "real", color = "blue")
        plt.plot(self.t[:-self.Nh],self.ra_r[0,:-self.Nh],label = "ref.", linestyle='dashed', color = "blue")
        plt.ylabel('X (m)')
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.t[:-self.Nh],self.X[7,:-self.Nh],label = "real", color = "blue")
        plt.plot(self.t[:-self.Nh],self.ra_r[1,:-self.Nh],label = "ref.", linestyle='dashed', color = "blue")
        plt.ylabel("Y (m)")
        plt.grid()
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.t[:-self.Nh],self.X[8,:-self.Nh],label = "real", color = "blue")
        plt.plot(self.t[:-self.Nh],self.ra_r[2,:-self.Nh],label = "ref.", linestyle='dashed', color = "blue")
        plt.ylabel("Z (m)")
        plt.xlabel('t (s)')
        plt.legend()
        plt.grid()
        if self.save:
            plt.savefig(f"{self.figures_path}/positions_plot_{self.id}.png")

    def plot_error_position(self):
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[:-self.Nh],self.X[6,:-self.Nh]-self.ra_r[0,:-self.Nh],label = "x")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.t[:-self.Nh],self.X[7,:-self.Nh]-self.ra_r[1,:-self.Nh],label = "y")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.t[:-self.Nh],self.X[8,:-self.Nh]-self.ra_r[2,:-self.Nh],label = "z")
        plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/error_position_plot_{self.id}.png")

    def plot_error_velocity(self):
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.title("Agent " + self.id)
        plt.plot(self.t[:-self.Nh],self.X[3,:-self.Nh]-self.va_r[0,:-self.Nh],label = "u")
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(self.t[:-self.Nh],self.X[4,:-self.Nh]-self.va_r[1,:-self.Nh],label = "v")
        plt.legend()
        plt.subplot(3, 1, 3)
        plt.plot(self.t[:-self.Nh],self.X[5,:-self.Nh]-self.va_r[2,:-self.Nh],label = "w")
        plt.legend()
        if self.save:
            plt.savefig(f"{self.figures_path}/error_velocity_plot_{self.id}.png")


    def save_to_csv(self):
        # Save va_body to a CSV file
    # Convert all elements in pid.Car to Euler angles and save to a CSV file
        euler_angles = []
        for i in range(self.N):
            euler = R.from_matrix(self.Car[:, :, i]).as_euler('zyx', degrees=True)
            euler_angles.append(euler)

        with open("desired_euler_angles.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Roll (deg)", "Pitch (deg)", "Yaw (deg)"])
            writer.writerows(euler_angles)

        # Multiply every element of self.Car by self.va_r and save to another CSV file
        car_va_r_product = []
        for i in range(self.N):
            product = self.Car[:, :, i] @ self.va_r[:, i]
            car_va_r_product.append(product)

        with open("car_va_r_product.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Z"])
            writer.writerows(car_va_r_product)

        # Save self.wr_r to another CSV file
        with open("desired_wr_r.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Omega_p", "Omega_q", "Omega_r"])
            writer.writerows(self.wr_r.T)

        # Save all elements in self.ra_r to a CSV file
        with open("desired_positions.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["X", "Y", "Z"])
            writer.writerows(self.ra_r.T)
if __name__ == "__main__":
    pid = PID_fixed_wing()
    for i in range(pid.N-1):
        pid.references(i)
        
    # pid.plot_force()
    # plt.show()
    # pid.plot_references()
    # pid.plot_acceleration()
    # plt.show()
    pid.control()

    pid.plot_angles()
    pid.plot_3D()
    pid.plot_erros()
    # pid.plot_error_f()
    pid.plot_controls()
    pid.plot_velocity()
    # pid.plot_angles()
    pid.plot_positions()
    plt.show()
    print("done")