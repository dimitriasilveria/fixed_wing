import numpy as np
from utils import R3_so3, dX_to_dXi, references, so3_R3, SE3_se3_back
from scipy.linalg import expm, logm
from numpy.linalg import matrix_rank
from scipy.linalg import block_diag
from pyfly.pyfly import PyFly
import control as ctrl
import pandas as pd
# from scipy.signal import place_poles
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import os
from icecream import ic
import csv

#Initiating constants##############################################
class PID_fixed_wing():
    def __init__(self):
        #fixed wing model
        self.fixed_wing = PyFly("/home/dimitria/fixed_wing/pyfly/pyfly/pyfly_config.json", "/home/dimitria/fixed_wing/pyfly/pyfly/x8_param.mat")
        self.fixed_wing.seed(0)
        self.fixed_wing.reset(state={"roll": 0, "pitch": 0})

        self.qsi = np.zeros((9,1)) 
        self.n_agents = 1
        self.mb = self.fixed_wing.params["mass"]
        self.g = self.fixed_wing.g
        self.ga = np.array([0, 0, -self.g])

        self.Jb_Bz = np.diag([0.0112,0.01123,0.02108]) 
        self.D = np.diag([0.605,0.44,0.275]) 
        self.E = np.diag([0.05,0.05,0.05]) 
        self.F = np.diag([0.1,0.1,0.1]) 
        self.K_w = np.diag([5,5,5])*1 
        self.K_i = np.diag([3,3,3])*1 
        self.I3 = np.array([0, 0, 1])
        
        self.r = 200
        self.T_p = 46
        # self.r = 100
        # self.T_p = 25
        self.t_max = 2#/5
        self.t_min = 0
        
        self.T = self.fixed_wing.dt
        self.N = int((self.t_max-self.t_min)/self.T) #number of points
        ic(self.N)
        #reference trajectories###################################################

        #circle
        # self.ra_r = np.vstack(self.r*np.cos(2*np.pi*self.t/self.T_p), self.r*np.sin(2*np.pi*self.t/self.T_p), [0.6*np.ones_like(self.t)])  #reference position
        # self.va_r = np.vstack((2*np.pi/self.T_p)*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), [(2*np.pi/self.T_p)*self.r*np.cos(2*np.pi*self.t/self.T_p), 0.6*np.zeros_like(self.t)]) #reference linear velocity
        # self.va_r_dot = np.vstack([(2*np.pi/self.T_p)**2*(-self.r*np.cos(2*np.pi*self.t/self.T_p)), (2*np.pi/self.T_p)**2*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), np.zeros_like(self.t)]) #reference linear acceleration 
        self.wr_r = np.zeros((3,self.N))  #reference angular velocity
        self.Car = np.zeros((3,3,self.N))  #reference attitude that converts from body fame to inertial
        #self.Car[:,:,0] = np.eye(3) #eul2rotm([pi*0.5 0 0])*Ca_r(:,:,1) #
        self.z_w = np.array([0,0,1])
        self.f_r= np.zeros((3,self.N))  #reference force
        # self.f_A_r = np.zeros((3,self.N))  #reference aerodynamics force
        #controller and state-space matrices ###################################
        self.A = np.zeros((9,9,self.N))
        self.B = np.zeros((9,6,self.N)) #check this

        self.t = np.linspace(self.t_min,self.t_max,self.N) #time
        self.ra_r = np.vstack((self.r*np.cos(2*np.pi*self.t/self.T_p), self.r*np.sin(2*np.pi*self.t/self.T_p), -1*np.ones_like(self.t)))  #reference position
        self.va_r = np.vstack(((2*np.pi/self.T_p)*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), (2*np.pi/self.T_p)*self.r*np.cos(2*np.pi*self.t/self.T_p), 0.6*np.zeros_like(self.t))) #reference linear velocity
        self.va_r_dot = np.vstack(((2*np.pi/self.T_p)**2*(-self.r*np.cos(2*np.pi*self.t/self.T_p)), (2*np.pi/self.T_p)**2*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), np.zeros_like(self.t))) #reference linear acceleration 
        self.c1 = 0.10
        self.va_r_dot_body = np.zeros((3,self.N))

        p = np.linspace(-1, -9, 9)
        self.p_disc = np.exp(p*self.T)
        self.K_pid = np.zeros((6,9,self.N))
        #actual trajectory and errors ###########################################
        self.Cab = np.zeros((3,3,self.N))
        self.Cab[:,:,0] = np.eye(3) #eul2rotm([pi*0.5 0 0])*Ca_r(:,:,1) #
        self.f = np.zeros((3,self.N))  #actual force
        # self.f_A = np.zeros((3,self.N))  #actual aerodynamics force
        self.wb_b_cont = np.zeros((3,self.N)) #angular velocity input
        self.d_Xi = np.zeros((9,self.N))  #error
        self.dU = np.zeros((6,self.N))
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


       
        # self.d_Xi[9:12,0] = self.c1*self.d_Xi[6:9,0] + self.d_Xi[3:6,0] 




    def references(self,i):
        
            #attitude = R.from_matrix(Car[:,:,i]).as_quat()
        v_norm = self.va_r[:,i]/np.linalg.norm(self.va_r[:,i])
        v_xy_norm = np.linalg.norm(self.va_r[0:2,i])
        # a_c = v_norm**2/self.r
        #roll (phi)
        phi = np.arctan2(-np.linalg.norm(self.va_r[:,i])**2,self.g*self.r)
        #correcting the acceleration
        # self.va_r_dot[:,i] = self.va_r_dot[:,i] - np.array([0,0,self.g - (self.g/np.cos(phi))]) 
        #yaw (psi)

        psi = np.arctan2(v_norm[1],v_norm[0])
        #pitch (theta)
        theta = np.arctan2(self.va_r[2,i],v_xy_norm)
        #ccentripetal acceleration component
        # ac = np.cross(self.va_r[0:2,i],self.va_r_dot[0:2,i])/v_xy_norm


        if i == 0:
            ic(np.rad2deg(phi))
            ic("minimum lift: ", self.mb*self.g/np.cos(phi))


            # Rotation matrices using Z-Y-X intrinsic rotations
        Rz = np.array([
            [np.cos(psi), np.sin(psi), 0],
            [-np.sin(psi), np.cos(psi),  0],
            [0,           0,            1]
        ])

        Ry = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0,             1, 0],
            [np.sin(theta),0, np.cos(theta)]
        ])

        Rx = np.array([
            [1, 0,          0],
            [0, np.cos(phi), np.sin(phi)],
            [0, -np.sin(phi),  np.cos(phi)]
        ])

        Rx_180 = np.array([
            [1, 0,          0],
            [0, np.cos(np.pi), -np.sin(np.pi)],
            [0, np.sin(np.pi),  np.cos(np.pi)]])


        self.Car[:,:,i] =  (Rx @ Ry @ Rz) #@Rx_180
        

            

        # self.Car[:,:,i] = np.hstack((x_B.reshape(3,1), y_B.reshape(3,1), z_B.reshape(3,1)))#@R.from_euler('xyz', [np.pi/100, 0, 0]).as_matrix()
        if i > 0 :
            self.wr_r[:,i] = so3_R3(logm(self.Car[:,:,i-1].T@self.Car[:,:,i]))/self.T
            v_body = self.Car[:,:,i].T@self.va_r[:,i]
            a_body  = self.Car[:,:,i].T@self.va_r_dot[:,i]
            self.va_r_dot_body[:,i] = a_body
            wr_r_inertial = self.Car[:,:,i]@self.wr_r[:,i]
            # self.f_r[:,i] = self.mb*(self.va_r_dot[:,i] - self.g*self.z_w  + np.cross(wr_r_inertial,self.va_r[:,i]))
            self.f_r[:,i] = self.mb*(a_body - self.g*self.Car[:,:,i].T@self.z_w  + np.cross(self.wr_r[:,i],v_body)) 
            # self.f_r[:,i] = self.Car[:,:,i]@self.f_r[:,i]
            # self.f_r[2,i] = -self.mb*self.g/np.cos(phi)
            # self.f_r[:,i] = self.Car[:,:,i].T@(self.mb*self.va_r_dot[:,i] - self.mb*self.g*self.z_w) 
        else:
            self.f_r[:,i] = self.Car[:,:,i].T@(self.mb*self.va_r_dot[:,i] - self.mb*self.g*self.z_w)  
            # if np.linalg.det(self.Car[:,:,i]) != 0 and np.linalg.det(self.Car[:,:,i+1]) != 0:
                
            # else:
            #     self.wr_r[:,i] = np.zeros(3)

    def calc_A_and_B(self,i):

        self.A[3:6,0:3,i]   = (1/self.mb)*(-R3_so3(self.f[:,i])) 
        self.A[3:6,3:6,i]   = -R3_so3(self.wr_r[:,i])  
        self.A[6:9,3:6,i]   = np.eye(3) 
        self.A[6:9,6:9,i]   = -R3_so3(self.wr_r[:,i]) 
        # self.A[9:12,3:6,i]  = np.eye(3) 
        # self.A[9:12,6:9,i]  = self.c1*np.eye(3) 
        # self.B[5,0,i] = 1/self.mb 
        self.B[3:6, 0:3,i] = np.eye(3)/self.mb
        self.B[0:3, 3:6,i]  = np.eye(3)  
        self.B[3:6, 3:6,i]  = R3_so3(self.Car[:,:,i].T@self.va_r[:,i])/self.mb
        #self.B[7, 0, i] = 1
        self.A[:,:,i] = np.eye(9) + self.T * self.A[:,:,i] 
        self.B[:,:,i] = self.T*self.B[:,:,i]#+ 1e-5 * np.random.randn(*self.B[:,:,i].shape)

        # Co = self.B[:,:,i]
        # for j in range(1, self.A[:,:,i].shape[0]):
        #     Co = np.hstack((Co, np.linalg.matrix_power(self.A[:,:,i], j) @ self.B[:,:,i]))
        # rank_Co = matrix_rank(Co)


        # print(f"Controllability matrix rank: {rank_Co}")
        # print(f"Expected rank (should be {self.A[:,:,i].shape[0]}): {self.A[:,:,i].shape[0]}")

        # eig_A = np.linalg.eigvals(self.A[:,:,i])
        # print("Eigenvalues of A:", eig_A)


    def calc_K_pid(self,i):
        self.K_pid[:,:,i] = ctrl.place(self.A[:,:,i],self.B[:,:,i],self.p_disc)#, method='YT')

    def control(self):

        for i in range(2):
            self.X[3:6,i] = self.va_r[:,i] 
            self.X[6:9,i] = self.ra_r[:,i] 
            self.fixed_wing.state["omega_p"].value = self.wr_r[0,i]
            self.fixed_wing.state["omega_q"].value = self.wr_r[1,i]
            self.fixed_wing.state["omega_r"].value = self.wr_r[2,i]
            self.fixed_wing.state["position_n"].value = self.ra_r[0,i]
            self.fixed_wing.state["position_e"].value = self.ra_r[1,i]
            self.fixed_wing.state["position_d"].value = self.ra_r[2,i]
            va_body = self.Car[:,:,i].T@self.va_r[:,i]

            self.fixed_wing.state["velocity_u"].value = va_body[0]
            self.fixed_wing.state["velocity_v"].value = va_body[1]
            self.fixed_wing.state["velocity_w"].value = va_body[2]
            self.fixed_wing.state["attitude"].set_value(R.from_matrix(self.Car[:,:,i]).as_quat())
            self.Cab[:,:,i] = R.from_quat(self.fixed_wing.state["attitude"].value).as_matrix()
            self.dC[:,:,i] = self.Cab[:,:,i].T@self.Car[:,:,i]
            self.d_v = self.Cab[:,:,i].T@(self.va_r[:,i] - self.X[3:6,i]) 
            self.d_r = self.Cab[:,:,i].T@(self.ra_r[:,i] - self.X[6:9,i]) 
            self.d_Xi[0:9,i] = dX_to_dXi(self.dC[:,:,i],self.d_v,self.d_r) 

        for i in range(1,self.N-1):
            if i == 100:
                ic()
            self.calc_K_pid(i)
            # ic(self.d_Xi[0:9,i])
            # input()
            self.dU[:,i] = -self.K_pid[:,:,i]@self.d_Xi[:,i]
            # self.dU[2,i] *= -1
            # self.dU[0:3,i] =np.clip(self.dU[0:3,i],-10,10)
            # self.dU[3:6,i] =np.clip(self.dU[3:6,i],-np.pi/3,np.pi/3)
            self.f[:,i] = self.f_r[:,i] - self.dU[0:3,i]
            self.wb_b_cont[:,i] = self.dC[:,:,i]@self.wr_r[:,i] - self.dU[3:6,i]
            #self.wb_b_cont[:,i] = np.clip(self.wb_b_cont[:,i],-np.pi/3,np.pi/3)
            attitude = R.from_matrix(self.Cab[:,:,i]).as_quat()

            controls = self.fixed_wing.calc_control(attitude, self.Cab[:,:,i].T@self.X[3:6,i], self.wb_b_cont[:,i], self.f[:,i]) #should I use Cab or Car? 
            #printing all the states

            _, _ = self.fixed_wing.step(controls)

            va_body = self.Car[:,:,i+1].T@self.va_r[:,i+1]
            # Save va_body to a CSV file

            # self.fixed_wing.state["velocity_u"].value = va_body[0]
            # self.fixed_wing.state["velocity_v"].value = va_body[1]
            # self.fixed_wing.state["velocity_w"].value = va_body[2]
            # self.fixed_wing.state["attitude"].set_value(R.from_matrix(self.Car[:,:,i+1]).as_quat())
            # self.fixed_wing.state["omega_p"].value = self.wr_r[0,i+1]
            # self.fixed_wing.state["omega_q"].value = self.wr_r[1,i+1]
            # self.fixed_wing.state["omega_r"].value = self.wr_r[2,i+1]

            
            # Convert self.Car[:,:,i+1] to Euler angles


            #updating the states
            self.Cab[:,:,i+1] = R.from_quat(self.fixed_wing.state["attitude"].value).as_matrix()
            self.X[0,i+1] = self.fixed_wing.state["omega_p"].value
            self.X[1,i+1] =self.fixed_wing.state["omega_q"].value
            self.X[2,i+1] =self.fixed_wing.state["omega_r"].value
            self.X[6,i+1] =self.fixed_wing.state["position_n"].value
            self.X[7,i+1] =self.fixed_wing.state["position_e"].value
            self.X[8,i+1] =self.fixed_wing.state["position_d"].value
            va_inertial = self.Cab[:,:,i+1]@np.array([self.fixed_wing.state["velocity_u"].value,self.fixed_wing.state["velocity_v"].value,self.fixed_wing.state["velocity_w"].value])
            self.X[3,i+1] = va_inertial[0]
            self.X[4,i+1] = va_inertial[1]
            self.X[5,i+1] = va_inertial[2]
            #printing the next desired values           
            
            self.dC[:,:,i+1] = self.Cab[:,:,i+1].T@self.Car[:,:,i+1]
            # self.dC[:,:,i+1] = self.Cab[:,:,i+1].T@self.Car[:,:,i+1]
            self.d_v = self.Cab[:,:,i+1].T@(self.va_r[:,i+1] - self.X[3:6,i+1]) 
            self.d_r = self.Cab[:,:,i+1].T@(self.ra_r[:,i+1] - self.X[6:9,i+1]) 
            self.d_Xi[0:9,i+1] = dX_to_dXi(self.dC[:,:,i+1],self.d_v,self.d_r) 
            # self.d_Xi[9:12,i+1] = self.c1@self.d_Xi[6:9,i+1] + self.d_Xi[3:6,i+1] 

            self.abs_phi[0,i+1] =np.linalg.norm(self.d_Xi[0:3,i+1]) 
            self.abs_r[0,i+1] =  np.linalg.norm(self.d_Xi[6:9,i+1]) 
            self.abs_v[0,i+1] =  np.linalg.norm(self.d_Xi[3:6,i+1]) 
            self.abs_f[0,i] = np.linalg.norm(self.dU[0:3,i]) 
            self.abs_w[0,i] =np.linalg.norm(self.dU[3:6,i]) 

            ic(i)

    def plot_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.X[6,:],self.X[7,:],self.X[8,:],label = "real trajectory")
        ax.plot(self.ra_r[0,:],self.ra_r[1,:],self.ra_r[2,:],label = "reference trajectory")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # ax.set_zlim(-1,0.1)
        # ax.set_ylim(0,20)
        # ax.set_xlim(190,200)
        ax.legend()
        # plt.show()
            #would f_T be the same as f_T_r depending on what attitude I use?
    def plot_erros(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t,self.abs_phi[0,:],label = "phi error")
        ax.plot(self.t,self.abs_r[0,:],label = "r error")
        ax.plot(self.t,self.abs_v[0,:],label = "v error")
        ax.plot(self.t,self.abs_w[0,:],label = "w error")
        ax.set_xlabel('time')
        ax.set_ylabel('error')
        ax.legend()
        # plt.show()
    def plot_error_f(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t,self.abs_f[0,:],label = "f error")
        ax.set_xlabel('time')
        ax.set_ylabel('error')
        ax.legend()
        # plt.show()

    def plot_acceleration(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t,self.va_r_dot_body[0,:],label = "a_x")
        plt.plot(self.t,self.va_r_dot_body[1,:],label = "a_y")
        plt.plot(self.t,self.va_r_dot_body[2,:],label = "a_z")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.t,self.va_r[0,:],label = "v_x")
        plt.plot(self.t,self.va_r[1,:],label = "v_y")
        plt.plot(self.t,self.va_r[2,:],label = "v_z")
        plt.legend()
        # plt.show()
    def plot_controls(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t,self.f[0,:],label = "f_x")
        plt.plot(self.t,self.f[1,:],label = "f_y")
        plt.plot(self.t,self.f[2,:],label = "f_z")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.t,self.wb_b_cont[0,:],label = "omega_p")
        plt.plot(self.t,self.wb_b_cont[1,:],label = "omega_q")
        plt.plot(self.t,self.wb_b_cont[2,:],label = "omega_r")
        plt.legend()
        # plt.show()

    def plot_references(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t,self.f_r[0,:],label = "f_x")
        plt.plot(self.t,self.f_r[1,:],label = "f_y")
        plt.plot(self.t,self.f_r[2,:],label = "f_z")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.t,self.wr_r[0,:],label = "omega_p")
        plt.plot(self.t,self.wr_r[1,:],label = "omega_q")
        plt.plot(self.t,self.wr_r[2,:],label = "omega_r")
        plt.legend()

    def plot_velocity(self):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t[1:],self.X[3,1:],label = "u")
        plt.plot(self.t[1:],self.X[4,1:],label = "v")
        plt.plot(self.t[1:],self.X[5,1:],label = "w")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.t[1:],self.va_r[0,1:],label = "u_r")
        plt.plot(self.t[1:],self.va_r[1,1:],label = "v_r")
        plt.plot(self.t[1:],self.va_r[2,1:],label = "w_r")
        plt.legend()
        # plt.show()

    def save_to_csv(self):
        # Save va_body to a CSV file
    # Convert all elements in pid.Car to Euler angles and save to a CSV file
        euler_angles = []
        for i in range(self.N):
            euler = R.from_matrix(self.Car[:, :, i]).as_euler('xyz', degrees=True)
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
    for i in range(pid.N):
        pid.references(i)
        pid.calc_A_and_B(i)
    # pid.plot_references()
    # pid.plot_acceleration()
    # plt.show()
    pid.control()


    pid.plot_3D()
    pid.plot_erros()
    pid.plot_error_f()
    pid.plot_controls()
    pid.plot_velocity()
    plt.show()
    print("done")