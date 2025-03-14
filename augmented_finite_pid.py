import numpy as np
from utils import R3_so3, dX_to_dXi, references, so3_R3, SE3_se3_back
from scipy.linalg import expm, logm
from numpy.linalg import matrix_rank
from scipy.linalg import block_diag
from pyfly.pyfly import PyFly
import control as ctrl
# from scipy.signal import place_poles
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from icecream import ic

#Initiating constants##############################################
class PID_fixed_wing():
    def __init__(self):
        #fixed wing model
        self.fixed_wing = PyFly("/home/dimitria/fixed_wing/pyfly/pyfly/pyfly_config.json", "/home/dimitria/fixed_wing/pyfly/pyfly/x8_param.mat")
        self.fixed_wing.seed(0)
        self.fixed_wing.reset(state={"roll": -0.5, "pitch": 0.15})
        self.t_max = 1
        self.N = self.t_max*10 #number of points
        self.T = (self.t_max-0)/self.N 
        self.T_p = 30 
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
        self.t = np.linspace(0,self.t_max,self.N) #time

        #reference trajectories###################################################
        self.r = 1 
        #circle
        # self.ra_r = np.vstack(self.r*np.cos(2*np.pi*self.t/self.T_p), self.r*np.sin(2*np.pi*self.t/self.T_p), [0.6*np.ones_like(self.t)])  #reference position
        # self.va_r = np.vstack((2*np.pi/self.T_p)*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), [(2*np.pi/self.T_p)*self.r*np.cos(2*np.pi*self.t/self.T_p), 0.6*np.zeros_like(self.t)]) #reference linear velocity
        # self.va_r_dot = np.vstack([(2*np.pi/self.T_p)**2*(-self.r*np.cos(2*np.pi*self.t/self.T_p)), (2*np.pi/self.T_p)**2*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), np.zeros_like(self.t)]) #reference linear acceleration 
        self.wr_r = np.zeros((3,self.N))  #reference angular velocity
        self.Car = np.zeros((3,3,self.N))  #reference attitude
        #self.Car[:,:,0] = np.eye(3) #eul2rotm([pi*0.5 0 0])*Ca_r(:,:,1) #
        self.z_w = np.array([0,0,-1])
        self.f_r= np.zeros((3,self.N))  #reference force
        # self.f_A_r = np.zeros((3,self.N))  #reference aerodynamics force
        #controller and state-space matrices ###################################
        self.A = np.zeros((9,9,self.N))
        self.B = np.zeros((9,6,self.N)) #check this
        self.ra_r = np.vstack((self.r*np.cos(2*np.pi*self.t/self.T_p), self.r*np.sin(2*np.pi*self.t/self.T_p), -0.6*np.ones_like(self.t)))  #reference position
        self.va_r = np.vstack(((2*np.pi/self.T_p)*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), (2*np.pi/self.T_p)*self.r*np.cos(2*np.pi*self.t/self.T_p), 0.6*np.zeros_like(self.t))) #reference linear velocity
        self.va_r_dot = np.vstack(((2*np.pi/self.T_p)**2*(-self.r*np.cos(2*np.pi*self.t/self.T_p)), (2*np.pi/self.T_p)**2*(-self.r*np.sin(2*np.pi*self.t/self.T_p)), np.zeros_like(self.t))) #reference linear acceleration 
        self.c1 = 0.10
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
        self.dC[:,:,0] = self.Cab[:,:,0].T@self.Car[:,:,0] 
        self.abs_phi = np.zeros((1,self.N))
        self.abs_r = np.zeros((1,self.N))
        self.abs_v = np.zeros((1,self.N))
        self.abs_f = np.zeros((1,self.N))
        self.abs_f_a = np.zeros((1,self.N))
        self.abs_w = np.zeros((1,self.N))

        self.X = np.zeros((9,self.N)) 
        self.X[3:6,0] = self.va_r[:,0] 
        self.X[6:9,0] = self.ra_r[:,0] 
        self.fixed_wing.state["omega_p"].value = self.wr_r[0,0]
        self.fixed_wing.state["omega_q"].value = self.wr_r[1,0]
        self.fixed_wing.state["omega_r"].value = self.wr_r[2,0]
        self.fixed_wing.state["position_n"].value = self.ra_r[0,0]
        self.fixed_wing.state["position_e"].value = self.ra_r[1,0]
        self.fixed_wing.state["position_d"].value = self.ra_r[2,0]
        self.fixed_wing.state["velocity_u"].value = self.va_r[0,0] 
        self.fixed_wing.state["velocity_v"].value = self.va_r[1,0] 
        self.fixed_wing.state["velocity_w"].value = self.va_r[2,0] 
        self.d_v = self.Cab[:,:,0].T@(self.va_r[:,0] - self.X[3:6,0]) 
        self.d_r = self.Cab[:,:,0].T@(self.ra_r[:,0] - self.X[6:9,0]) 
        self.d_Xi[0:9,0] = dX_to_dXi(self.dC[:,:,0],self.d_v,self.d_r) 
        
        # self.d_Xi[9:12,0] = self.c1*self.d_Xi[6:9,0] + self.d_Xi[3:6,0] 




    def references(self,i):
            #attitude = R.from_matrix(Car[:,:,i]).as_quat()
        v_norm = self.va_r[:,i]/np.linalg.norm(self.va_r[:,i])
        v_xy_norm = np.linalg.norm(self.va_r[0:2,i])

        #yaw (psi)
        psi = np.arctan2(v_norm[1],v_norm[0])
        #pitch (theta)
        theta = np.arctan2(-self.va_r[2,i],v_xy_norm)

        #ccentripetal acceleration component
        ac = np.cross(self.va_r[0:2,i],self.va_r_dot[0:2,i])/v_xy_norm

        #roll (phi)
        phi = np.arctan2(ac,self.g)
            # Rotation matrices using Z-Y-X intrinsic rotations
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
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



        self.Car[:,:,i] =  Rz @ Ry @ Rx
        # norm_aux = np.linalg.norm(self.va_r_dot[:,i]-(self.va_r_dot[:,i]@self.va_r[:,i])*self.va_r[:,i]) 
        # if (norm_x_B != 0) and (norm_aux != 0):
        #     x_B = self.va_r[:,i]/norm_x_B
        #     R_ = norm_x_B**2/norm_aux
        #     phi = np.arctan2(norm_x_B**2,(self.g*R_))
        #     z_intermediate = self.z_w - (self.z_w@x_B)*x_B
        #     norm_z_intermediate = np.linalg.norm(z_intermediate)
        #     if norm_z_intermediate != 0:
        #         z_intermediate = z_intermediate/norm_z_intermediate
        #         z_B = np.cos(phi)*z_intermediate + np.sin(phi)*(np.cross(x_B,z_intermediate))
        #         z_B = z_B/np.linalg.norm(z_B)
        #         y_B = np.cross(z_B,x_B)
        #         y_B = y_B/np.linalg.norm(y_B)
        #     else:
        #         y_B = np.zeros(3)
        #         z_B = np.zeros(3)
        # else:
        #     x_B = np.zeros(3)
        #     z_B = np.zeros(3)
        #     y_B = np.zeros(3)
            

        # self.Car[:,:,i] = np.hstack((x_B.reshape(3,1), y_B.reshape(3,1), z_B.reshape(3,1)))#@R.from_euler('xyz', [np.pi/100, 0, 0]).as_matrix()
        if i > 0 :
            self.wr_r[:,i] = so3_R3(logm(self.Car[:,:,i-1].T@self.Car[:,:,i]))/self.T
            self.f_r[:,i] = self.Car[:,:,i]@(self.mb*self.va_r_dot[:,i] - self.mb*self.g*self.z_w)
            ic(self.f_r[:,i])
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
        self.references(0)
        attitude = R.from_matrix(self.Car[:,:,0]).as_quat()
        omega = self.wr_r[:,0]
        vel = self.va_r[:,0]
        controls = np.zeros(4)
        # f = self.fixed_wing.reference_forces(attitude, omega, vel, controls)

        # self.f[:,0] = f
        for i in range(0,self.N-1):
            self.references(i+1)
            self.calc_A_and_B(i)
            self.calc_K_pid(i)
            # ic(self.d_Xi[0:9,i])
            # input()
            self.dU[:,i] = -self.K_pid[:,:,i]@self.d_Xi[:,i]
            #self.dU[0:3,i] =np.clip(self.dU[0:3,i],-1,1)
            self.f[:,i] = self.f_r[:,i] - self.dU[0:3,i]
            # self.f_r[:,i+1] = self.f[:,i]
            self.wb_b_cont[:,i] = self.dC[:,:,i]@self.wr_r[:,i] - self.dU[3:6,i]
            #self.wb_b_cont[:,i] = np.clip(self.wb_b_cont[:,i],-np.pi/3,np.pi/3)
            attitude = R.from_matrix(self.Car[:,:,i]).as_quat()
            controls = self.fixed_wing.calc_control(attitude, self.va_r[:,i], self.wb_b_cont[:,i], self.f[:,i]) #should I use Cab or Car? 
            ic(controls)
            _, _ = self.fixed_wing.step(controls)
            #self.f_r[:,i+1] = self.fixed_wing.reference_forces(attitude, omega, vel, controls)
            self.X[0,i+1] = self.fixed_wing.state["omega_p"].value
            self.X[1,i+1] =self.fixed_wing.state["omega_q"].value
            self.X[2,i+1] =self.fixed_wing.state["omega_r"].value
            self.X[6,i+1] =self.fixed_wing.state["position_n"].value
            self.X[7,i+1] =self.fixed_wing.state["position_e"].value
            self.X[8,i+1] =self.fixed_wing.state["position_d"].value
            self.X[3,i+1] =self.fixed_wing.state["velocity_u"].value
            self.X[4,i+1] =self.fixed_wing.state["velocity_v"].value
            self.X[5,i+1] =self.fixed_wing.state["velocity_w"].value
            self.Cab[:,:,i+1] = R.from_quat(self.fixed_wing.state["attitude"].value).as_matrix()
            self.dC[:,:,i+1] = self.Cab[:,:,i+1].T@self.Car[:,:,i+1]
            self.d_v = self.Cab[:,:,i+1].T@(self.va_r[:,i+1] - self.X[3:6,i+1]) 
            self.d_r = self.Cab[:,:,i+1].T@(self.ra_r[:,i+1] - self.X[6:9,i+1]) 
            self.d_Xi[0:9,i+1] = dX_to_dXi(self.dC[:,:,i+1],self.d_v,self.d_r) 
            # self.d_Xi[9:12,i+1] = self.c1@self.d_Xi[6:9,i+1] + self.d_Xi[3:6,i+1] 

            self.abs_phi[0,i+1] =np.linalg.norm(self.d_Xi[0:3,i+1]) 
            self.abs_r[0,i+1] =  np.linalg.norm(self.d_Xi[3:6,i+1]) 
            self.abs_v[0,i+1] =  np.linalg.norm(self.d_Xi[6:9,i+1]) 
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
        ax.legend()
        # plt.show()
            #would f_T be the same as f_T_r depending on what attitude I use?
    def plot_erros(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.t,self.abs_phi[0,:],label = "phi error")
        ax.plot(self.t,self.abs_r[0,:],label = "r error")
        ax.plot(self.t,self.abs_v[0,:],label = "v error")
        ax.plot(self.t,self.abs_f[0,:],label = "f error")
        ax.plot(self.t,self.abs_w[0,:],label = "w error")
        ax.set_xlabel('time')
        ax.set_ylabel('error')
        ax.legend()
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

if __name__ == "__main__":
    pid = PID_fixed_wing()
    pid.control()
    pid.plot_3D()
    pid.plot_erros()
    pid.plot_controls()
    plt.show()
    print("done")