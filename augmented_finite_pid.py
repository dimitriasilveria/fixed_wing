import numpy as np
from utils import R3_so3, dX_to_dXi, references, so3_R3, SE3_se3_back
from scipy.linalg import expm, logm
from pyfly.pyfly import PyFly
import control as ctrl
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm

#Initiating constants##############################################
class PID_fixed_wing():
    def __init__(self):
        self.t_max = 30
        self.N = self.t_max*1000  #number of points
        self.T = (self.t_max-0)/self.N 
        self.T_p = 30 
        self.qsi = np.zeros((9,1)) 
        self.n_agents = 1
        self.mb = 1 
        self.g = 9.8 
        self.ga = np.array([0, 0, -self.g])

        self.Jb_Bz = np.diag([0.0112,0.01123,0.02108]) 
        self.D = np.diag([0.605,0.44,0.275]) 
        self.E = np.diag([0.05,0.05,0.05]) 
        self.F = np.diag([0.1,0.1,0.1]) 
        self.K_w = np.diag([5,5,5])*1 
        self.K_i = np.diag([3,3,3])*1 
        self.I3 = np.array([[0], [0], [1]])
        self.t = np.linspace(0,self.t_max,self.N) #time


        #reference trajectories###################################################
        self.r = 1 
        #circle
        self.ra_r = np.array([[self.r*np.cos(2*np.pi*self.t/self.T_p)], [self.r*np.sin(2*np.pi*self.t/self.T_p)], [0.6*np.ones_like(self.t)]])  #reference position
        self.va_r = np.array([[(2*np.pi/self.T_p)*(-self.r*np.sin(2*np.pi*self.t/self.T_p))], [(2*np.pi/self.T_p)*self.r*np.cos(2*np.pi*self.t/self.T_p)], [0.6*np.zeros_like(self.t)]]) #reference linear velocity
        self.va_r_dot = np.array([[(2*np.pi/self.T_p)**2*(-self.r*np.cos(2*np.pi*self.t/self.T_p))], [(2*np.pi/self.T_p)**2*(-self.r*np.sin(2*np.pi*self.t/self.T_p))], [np.zeros_like(self.t)]]) #reference linear acceleration 
        self.wr_r = np.zeros((3,self.N,self.n_agents))  #reference angular velocity
        self.Car = np.zeros((3,3,self.N,self.n_agents))  #reference attitude
        self.z_w = np.array([0,0,1])
        self.f_T_r = np.zeros((1,self.N))  #reference thrust
        self.f_A_r = np.zeros((3,self.N))  #reference aerodynamics force
        #controller and state-space matrices ###################################
        self.A = np.zeros((12,12,self.N))
        self.B = np.zeros((12,4,self.N))
        self.B[5,0,:] = np.ones(self.N)/self.mb  #check this
        self.c1 = 1
        p = np.array([-5,-1.5, -1.1,-1.23,- 2.25,-5.5,-2.0,-1.0,-1.7])
        self.p_disc = np.exp(p*self.T)
        self.K_pid = np.zeros((4,12,self.N))
        #actual trajectory and errors ###########################################
        self.Cab = np.zeros((3,3,self.N))
        self.Cab[:,:,0] = np.eye(3) #eul2rotm([pi*0.5 0 0])*Ca_r(:,:,1) #
        self.f_T = np.zeros((1,self.N))  #actual thrust
        self.wb_b_cont = np.zeros((3,self.N)) #angular velocity input
        self.d_Xi = np.zeros((12,self.N))  #error
        self.dU = np.zeros((4,self.N))
        self.dC = np.zeros((3,3,self.N))
        self.d_v = np.zeros((3,self.N))
        self.d_r = np.zeros((3,self.N))
        dC[:,:,0] = self.Cab[:,:,0].T@self.Ca_r[:,:,0] 
        self.abs_phi = np.zeros((1,self.N))
        self.abs_r = np.zeros((1,self.N))
        self.abs_v = np.zeros((1,self.N))
        self.abs_f = np.zeros((1,self.N))
        self.abs_w = np.zeros((1,self.N))

        self.X = np.zeros((9,self.N)) 
        self.X[3:6,0] = self.va_r[:,0] 
        self.X[6:9,0] = self.ra_r[:,0] 
        self.d_v[:,0] = self.Cab[:,:,0].T@(self.va_r[:,0] - self.X[4:6,0]) 
        self.d_r[:,0] = self.Cab[:,:,0].T@(self.ra_r[:,0] - self.X[7:9,0]) 

        #fixed wing model
        self.fixed_wing = PyFly("/home/bitdrones/fixed_wing/pyfly/pyfly/pyfly_config.json", "/home/bitdrones/fixed_wing/pyfly/pyfly/x8_param.mat")
        self.fixed_wing.seed(0)
        self.fixed_wing.reset(state={"roll": -0.5, "pitch": 0.15})

    def references(self,i):
        for a in range(self.n_agents):

            #attitude = R.from_matrix(Car[:,:,i,a]).as_quat()
            norm_x_B = np.linalg.norm(self.va_r[:,a,i])
            norm_aux = np.linalg.norm(self.va_r_dot[:,a,i]-(self.va_r_dot[:,a,i]@self.va_r[:,a,i])*self.va_r[:,a,i]) 
            if (norm_x_B != 0) and (norm_aux != 0):
                x_B = self.va_r[:,a,i]/norm_x_B
                R = norm_x_B**2/norm_aux
                phi = np.arctan2(norm_x_B**2,(g*R))
                z_intermediate = self.z_w - (self.z_w@x_B)*x_B
                norm_z_intermediate = np.linalg.norm(z_intermediate)
                if norm_z_intermediate != 0:
                    z_intermediate = z_intermediate/norm_z_intermediate
                    z_B = np.cos(phi)*z_intermediate + np.sin(phi)*(np.cross(x_B,z_intermediate))
                    z_B = z_B/np.linalg.norm(z_B)
                    y_B = np.cross(z_B,x_B)
                    y_B = y_B/np.linalg.norm(y_B)
                else:
                    y_B = np.zeros(3)
                    z_B = np.zeros(3)
            else:
                x_B = np.zeros(3)
                z_B = np.zeros(3)
                y_B = np.zeros(3)
                

            self.Car[:,:,i+i,a] = np.hstack((x_B.reshape(3,1), y_B.reshape(3,1), z_B.reshape(3,1)))
            if np.linalg.det(self.Car[:,:,i,a]) != 0 and np.linalg.det(self.Car[:,:,i+1,a]) != 0:

                self.wr_r[:,i,a] = so3_R3(logm(np.linalg.inv(self.Car[:,:,i,a])@self.Car[:,:,i+1,a]))/self.dt
            else:
                self.wr_r[:,i,a] = np.zeros(3)

    def calc_A_and_B(self,i):

        self.A[3:6,0:3,i]   = (1/self.mb)*(-R3_so3((self.f_T_r[0,i]*self.I3+self.f_A_r[:,i]))) 
        self.A[3:6,3:6,i]   = -R3_so3(wr_r[:,i])  
        self.A[6:9,3:6,i]   = np.eye(3) 
        self.A[6:9,6:9,i]   = -R3_so3(wr_r[:,i]) 
        self.A[9:12,3:6,i]  = np.eye(3) 
        self.A[9:12,6:9,i]  = self.c1*np.eye(3) 
        self.B[0:3, 1:4,i]  = np.eye(3) 
        self.B[3:6, 1:4,i]  = self.Car[:,:,i].T@self.va_r[:,i]/mb

    
        self.A[:,:,i] = np.eye(12) + self.T * A[:,:,i]
        self.B[:,:,i] = self.T*self.B[:,:,i]

    def calc_K_pid(self,i):
        self.K_pid[:,:,i] = ctrl.place(self.A[:,:,i],self.B[:,:,i],self.p_disc)

    def control(self):
        
        attitude = R.from_matrix(self.Car[:,:,i]).as_quat()
        omega = self.wr_r[:,i]
        vel = self.va_r[:,i]
        controls = np.zeros(4)
        f_T, f_A = self.fixed_wing.reference_forces(self, attitude, omega, vel, controls)
        self.f_T_r[0,i] = f_T
        self.f_A_r[:,i] = f_A
        for i in range(0,self.N):
            self.dU[:,i] = -self.K_pid[:,:,i]@self.d_Xi[:,i]
            self.f_T[0,i] = self.f_T_r[0,i] - self.dU[0,i]
            self.f_T[0,i] = np.clip(self.f_T[0,i],0,20.44)
            self.wb_b_cont[:,i] = self.dC[:,:,i]@self.wr_r[:,i] - self.dU[1:4,i]
            self.wb_b_cont[:,i] = np.clip(self.wb_b_cont[:,i],-10*np.pi/3,10*np.pi/3)
            

wr_r = 0*va_r_dot  #reference angular velocity
fa_r = np.zeros(3,N)  #reference control force
f_T_r = np.zeros(1,N)  #reference thrust
wb_b_cont = np.zeros(3,N) 
f_T = np.zeros(1,N)  #actual thrust
dC = np.zeros(3,3,N) 
  #cossino matrix from the inertial frame to the reference
mb_r = np.zeros(3,N) 

#controller and state-space matrices ###################################


max1 = N 
max2 = N 
Q_f = 1/(1) 
Q_w = 1/(0.5) 

R_lqr = np.diag([Q_f,Q_w,Q_w,Q_w]) #1*eye(4) #

Q_v = 1/(0.1) 
Q_r = 1/(0.001) 
Q_phi = 1/(0.001) 
Q_aug = 1/0.01 
Q_lqr = np.diag([Q_phi,Q_phi,Q_phi,Q_v,Q_v,Q_v,Q_r,Q_r,Q_r,Q_aug,Q_aug,Q_aug]) #1*eye(9) #

P = np.zeros((12,12,N+1))
R = np.zeros((4,4,N+1))
K_lqr = np.zeros((4,12,N))
K_lqr2 = np.zeros((4,12,N))
K_p = np.zeros((4,12,N))
P[:,:,N] = Q_lqr[:,:,-1]


#actual trajectory and inputs ###########################################

#X(1:3,1) = wr_r(:,1) 
Cab = np.zeros((3,3,N)) 
Cab[:,:,0] = np.eye(3) #eul2rotm([pi*0.5 0 0])*Ca_r(:,:,1) #
d_Xi = np.zeros(12,N)  #error
dU = np.zeros((4,N)) 

d_v = Cab[:,:,0].T@(va_r[:,0] - X[4:6,0]) 
d_r = Cab[:,:,0].T@(ra_r[:,0] - X[7:9,0]) 
d_Xi[1:9,1] = dX_to_dXi(dC[:,:,0],d_v,d_r) 
abs_phi = np.zeros((1,N)) 
abs_r = abs_phi 
abs_v = abs_phi 
abs_f = abs_v 
abs_w = abs_f 
abs_phi[1,1] =np.linalg.norm(d_Xi[0:3,0]) 
abs_r[1,1] =np.linalg.norm(d_Xi[3:6,0]) 
abs_v[1,1] =np.linalg.norm(d_Xi[6:9,0]) 
e_w_int = 0 
wb_b = 0 
y_prev = np.zeros((6,3)) 
car = np.zeros((3,N)) 



fa_r, Ca_r, Wr_r = references(fixed_wing, va_r, va_r_dot, T,N,n_agents)

for i in range(N):
    A[3:6,0:3,i]   = (1/mb)*(-R3_so3(f_T_r[:,i]*I3)) 
    A[3:6,3:6,i]   = -R3_so3(wr_r[:,i])  
    A[6:9,3:6,i]   = np.eye(3) 
    A[6:9,6:9,i]   = -R3_so3(wr_r[:,i]) 
    A[9:12,3:6,i]= np.eye(3) 
    A[9:12,6:9,i]= c1*np.eye(3) 
    B[0:3, 1:4,i]  = np.eye(3) 
    B[3:6, 1:4,i]  = Ca_r[:,:,i].T@va_r[:,i]/mb

    
A = np.tile(np.eye(12), (N, 1, 1)).transpose(1, 2, 0) + T * A

B = T*B 
for i in range(N-1,0,-1):#calculating the controler gains

    R[:,:,i] = R_lqr + B[:,:,i-1].T@P[:,:,i]@B[:,:,i-1] 
    P[:,:,i-1] = A[:,:,i-1].T@(P[:,:,i]-P[:,:,i]@B[:,:,i-1]@np.linalg.inv(R[:,:,i])@B[:,:,i-1].T@P[:,:,i])@A[:,:,i-1] + Q_lqr 
    K_lqr[:,:,i-1] = np.linalg.inv(R[:,:,i])@B[:,:,i-1].T@P[:,:,i]@A[:,:,i-1] 
    #N = np.zeros(9,4) 
    #[K_lqr2(:,:,i),S_,P_] = dlqr(A(:,:,i-1),B(:,:,i-1),Q_lqr,R_lqr,N) 
#X(1:3,1) = dC(:,:,1)*wr_r(:,1) 
for i in range(0,N): #calculating the erros
    dU[:,i] = -K_lqr[:,:,i]@d_Xi[:,i] 
#     dU(1,i) = min(max(dU(1,i),-10),10) 
#     dU(2:4,i) = min(max(dU(2:4,i),-pi/2),pi/2) 
    #this part has to be modified############################################
    f_T[0,i] = f_T_r[0,i] - dU[0,i] 
    #20.44
    f_T[0,i] = np.clip(f_T[0,i],0,20.44) 
    #########################################################################
    wb_b_cont[:,i] = dC[:,:,i]@wr_r[:,i] - dU[1:4,i]
    #10*pi/3
    wb_b_cont[:,i] = np.clip(wb_b_cont[:,i],-10*np.pi/3,10*np.pi/3) 
    e_w = X[0:3,i]- wb_b_cont[:,i]
    e_w_int = e_w_int + e_w*T 

    mb_r[:,i] = E@Cab[:,:,i].T@X[3:6,i]+ F@X[0:3,i] - K_w@e_w - K_i@e_w_int 
    #evolving the system
    y_prev[1,:] = X[7:9,i].T 
    y_prev[2,:] = X[4:6,i].T 
    y_prev[3,:] = X[1:3,i].T 
    y_prev[4:6,:] = Cab[:,:,i]

    #this part needs to be modified############################################
    y_next = runge_kutta(f_T(1,i),mb_r(:,i),y_prev,T) 
    ###########################################################################
    X[0:3,i+1] = y_next[2,:].T #wb_b_cont(:,i) #wr_r(:,i+1) #
    X[3:6,i+1] = y_next[1,:].T #va_r(:,i+1) #
    X[6:9,i+1] = y_next[0,:].T #ra_r(:,i+1) #
    #Ri_next =y_next(4:6,:) 
    #Ri_next = 
    #[Q, ~] = qr(Ri_next) 
    Cab[:, :,i+1] = Cab[:,:,i]@expm(R3_so3(X[0:3,i+1])*T) #Ri_next 
    #Cab(:,:,i+1) = R #eul2rotm([y_next(3,1) y_next(2,1) y_next(1,1)]) #

    dC[:,:,i+1] = Cab[:,:,i+1].T@Ca_r[:,:,i+1]
    d_v = Cab[:,:,i+1].T@(va_r[:,i+1] - X[3:6,i+1]) 
    d_r = Cab[:,:,i+1].T@(ra_r[:,i+1] - X[6:9,i+1]) 
    d_Xi[0:9,i+1] = dX_to_dXi(dC[:,:,i+1],d_v,d_r) 
    d_Xi[9:12,i+1] = c1@d_Xi[6:9,i+1] + d_Xi[3:6,i+1] 

    abs_phi[0,i+1] =np.linalg.norm(d_Xi[0:3,i+1]) 
    abs_r[0,i+1] =  np.linalg.norm(d_Xi[3:6,i+1]) 
    abs_v[0,i+1] =  np.linalg.norm(d_Xi[6:9,i+1]) 
    abs_f[0,i] = np.abs(dU[0,i]) 
    abs_w[0,i] =np.linalg.norm(dU[1:4,i]) 




# fig = figure(4) 
# plot(t,va_r(1,:))
# hold on
# plot(t,va_r(2,:))
# hold off
# legend('X','Y')
# exportgraphics(fig,'3D_traj_proportional.png')
X_plot = [] 
C = [] 
for i=1:500:N
    X_plot = [X_plot  X(7:9,i)] 
    C =  [C  rotm2quat(Cab(:,:,i))] 
end
# fig = figure(4) 
# plotTransforms(X_plot, C, "FrameSize", 1)
# hold on
# #plotTransforms(ra_r(1:3,1:2000), rotm2quat(Ca_r(:,:,1:2000)), "FrameSize", 40)
# plot3(X(7,:),X(8,:),X(9,:),'k')
# hold off
# grid on
# axis equal
# xlabel('x - axis')  ylabel('y - axis')  zlabel('z - axis') 

# fig = figure(5) 
# plot(t,car(1,:))
# hold on
# plot(t,car(2,:))
# plot(t,car(3,:))
# hold off
# legend('yaw','pitch','roll')


function dydt = ode_system(y,ft,mbr,Cab)
    #ra = y(1,:)' 
    h = 0.0005 
    va = y(2,:)' 
    w = y(3,:)' 
    D = diag([0.605,0.44,0.275]) 
    E = diag([0.05,0.05,0.05]) 
    F = diag([0.1,0.1,0.1]) 
    I3 = [0 0 1]'  
    ga = [0 0 -9.8]' 
    mb = 1 
    Jb_Bz = diag([0.0112,0.01123,0.02108]) 
    Cab = Cab*expm(R3_so3(w*h)) 
    dydt = np.zeros(3,3) 
    dydt(1,:) = y(2,:) 
    dydt(2,:) = ga + (Cab*I3*ft/mb)-Cab*D*Cab'*va/mb  
    dydt(3,:)= inv(Jb_Bz)*(mbr - R3_so3(w)*Jb_Bz*w - E*Cab'*va-F*w) 
    #dydt(4:6,:) = Cab*R3_so3(w) 

end

function y_next = runge_kutta(ft,mbr,y_prev,dt)
    h = 0.0005 
    Cab = y_prev(4:6,:) 
    y_next = np.zeros(6,3) 
    M = dt/h 
    for i=1:M
        k1 = ode_system(y_prev(1:3,:),ft,mbr,Cab) 
        k2 = ode_system(y_prev(1:3,:) + 0.5*h*k1,ft,mbr,Cab) 
        k3 = ode_system(y_prev(1:3,:) + 0.5*h*k2,ft,mbr,Cab) 
        k4 = ode_system(y_prev(1:3,:) + k3*h,ft,mbr,Cab) 
        y_next(1:3,:) = y_prev(1:3,:) + (h/6)*(k1 + 2*k2 + 2*k3 + k4) 
        y_prev = y_next 
    end
end