from pyfly.pyfly import PyFly
from pyfly.pid_controller import PIDController
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import R3_so3, so3_R3
from icecream import ic
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from embedding_SO3_sim import Embedding
import math
#reference trajectory
t_max = 30
N = t_max*1  #number of points
T = (t_max-0)/N 

t = np.linspace(0,t_max,N) #time
g = 9.81 #gravity

dt = 0.1 # time step
n_agents = 1
w = 0.5
k_phi = 15
x = np.zeros((3,N,n_agents))
x_dot = np.zeros((3,N,n_agents))
x_dot_dot = np.zeros((3,N,n_agents))
Car = np.zeros((3,3,N))
Wr_r = np.zeros(3)
if n_agents >1:
    n_diff = int(math.factorial(n_agents) / (math.factorial(2) * math.factorial(n_agents-2)))
else:
    n_diff = 1
r = 191
T_p = 46 
distances = np.zeros((n_diff,N))
phi_diff =  np.zeros((n_diff,N))
ra_r = np.array([r*np.cos(2*np.pi*t/T_p), r*np.sin(2*np.pi*t/T_p), -np.ones_like(t)])  #reference position
va_r = np.array([(2*np.pi/T_p)*(-r*np.sin(2*np.pi*t/T_p)), (2*np.pi/T_p)*r*np.cos(2*np.pi*t/T_p), -np.zeros_like(t)] ) #reference linear velocity
va_r_dot = np.array([(2*np.pi/T_p)**2*(-r*np.cos(2*np.pi*t/T_p)), (2*np.pi/T_p)**2*(-r*np.sin(2*np.pi*t/T_p)), np.zeros_like(t)]) #reference linear acceleration
z_w = np.array([0,0,1])


for i in range(N-1):


        # Normalize velocity vector
        v_norm = va_r[:,i]/np.linalg.norm(va_r[:,i])
        v_xy_norm = np.linalg.norm(va_r[0:2,i])
        # a_c = v_norm**2/r
        #roll (phi)
        phi = np.arctan2(-np.linalg.norm(va_r[:,i])**2,g*r)
        #correcting the acceleration
        # va_r_dot[:,i] = va_r_dot[:,i] - np.array([0,0,g - (g/np.cos(phi))]) 
        #yaw (psi)

        psi = np.arctan2(v_norm[1],v_norm[0])
        #pitch (theta)
        theta = np.arctan2(va_r[2,i],v_xy_norm)
        #ccentripetal acceleration component
        # ac = np.cross(va_r[0:2,i],va_r_dot[0:2,i])/v_xy_norm


        if i == 0:
            ic(np.rad2deg(phi))


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
        # Rz_180 = np.array([
        #     [np.cos(-np.pi), -np.sin(-np.pi), 0],
        #     [np.sin(-np.pi), np.cos(-np.pi),  0],
        #     [0,           0,            1]
        # ])


        Car[:,:,i] =  (Rx @ Ry @ Rz).T
        if np.linalg.det(Car[:,:,i]) != 0 and np.linalg.det(Car[:,:,i+1]) != 0:

            Wr_r[:] = so3_R3(logm(np.linalg.inv(Car[:,:,i])@Car[:,:,i+1]))/dt
        else:
            Wr_r[:] = np.zeros(3)
        #yaw = R.from_matrix(Car[:,:,i,a]).as_euler('zyx')[0]
    # ic(X,yaw)
    # input()

# Initialisation de la figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Affichage des axes d'orientation
scale = 10.5  # Taille des axes
for i in range(N):
    # Matrice de rotation
    origin = ra_r[:,i]
    R_ = Car[:,:,i]
    # Vecteurs propres
    x_axis = R_[:,0] * scale
    y_axis = R_[:,1] * scale
    z_axis = R_[:,2] * scale



    # Affichage des vecteurs propres
    # Tracé des axes
    ax.quiver(*origin, *x_axis, color='r', linewidth=1, label="X local" if i == 0 else "")
    ax.quiver(*origin, *y_axis, color='g', linewidth=1, label="Y local" if i == 0 else "")
    ax.quiver(*origin, *z_axis, color='b', linewidth=1, label="Z local" if i == 0 else "")

# Légendes et affichage
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(-2-r, r+2)
ax.set_ylim(-2-r, r+2)
ax.set_xlim(-2-r, r+2)
ax.legend()

# fig = plt.figure()
# for i in range(n_diff):
#     plt.plot(distances[i,0:-1],label=f"Distance agent {i+1}")
# plt.ylabel("Distances (m)")
# plt.xlabel("Time (s)")
# plt.title("Distances between agents")
# plt.legend()
# fig = plt.figure()
# for i in range(n_diff):
#     plt.plot(np.rad2deg(phi_diff[i,0:-1]),label=f"Phase difference agent {i+1}")
# plt.title("Phase differences between agents")
# plt.ylabel("$\phi$ (degrees)")
# plt.xlabel("Time (s)")
# plt.legend()


plt.show()