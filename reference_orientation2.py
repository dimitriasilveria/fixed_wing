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
T_p = 30 
t = np.linspace(0,t_max,N) #time
g = 9.81 #gravity
r = 2 # radius
dt = 0.1 # time step
n_agents = 1
w = 0.5
k_phi = 15
x = np.zeros((3,N,n_agents))
x_dot = np.zeros((3,N,n_agents))
x_dot_dot = np.zeros((3,N,n_agents))
Car = np.zeros((3,3,N,n_agents))
Wr_r = np.zeros((3,n_agents))
if n_agents >1:
    n_diff = int(math.factorial(n_agents) / (math.factorial(2) * math.factorial(n_agents-2)))
else:
    n_diff = 1
distances = np.zeros((n_diff,N))
phi_diff =  np.zeros((n_diff,N))
ra_r = np.array([[r*np.cos(2*np.pi*t/T_p)], [r*np.sin(2*np.pi*t/T_p)], [0.6*np.ones_like(t)]])  #reference position
va_r = np.array([[(2*np.pi/T_p)*(-r*np.sin(2*np.pi*t/T_p))], [(2*np.pi/T_p)*r*np.cos(2*np.pi*t/T_p)], [0.6*np.zeros_like(t)]] ) #reference linear velocity
va_r_dot = np.array([[(2*np.pi/T_p)**2*(-r*np.cos(2*np.pi*t/T_p))], [(2*np.pi/T_p)**2*(-r*np.sin(2*np.pi*t/T_p))], [np.zeros_like(t)]]) #reference linear acceleration
z_w = np.array([0,0,1])


for i in range(N-1):


    for a in range(n_agents):
        X = np.arctan2(x_dot[1,i,a],x_dot[0,i,a])

        ca_1 = np.array([np.cos(X),np.sin(X),0]).T #auxiliar vector 
        #attitude = R.from_matrix(Car[:,:,i,a]).as_quat()
        norm_x_B = np.linalg.norm(va_r[:,a,i])
        norm_aux = np.linalg.norm(va_r_dot[:,a,i]-(va_r_dot[:,a,i]@va_r[:,a,i])*va_r[:,a,i]) 
        if (norm_x_B != 0) and (norm_aux != 0):
            x_B = va_r[:,a,i]/norm_x_B
            R = norm_x_B**2/norm_aux
            phi = np.arctan2(norm_x_B**2,(g*R))
            z_intermediate = z_w - (z_w@x_B)*x_B
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
            

        Car[:,:,i,a] = np.hstack((x_B.reshape(3,1), y_B.reshape(3,1), z_B.reshape(3,1)))
        if np.linalg.det(Car[:,:,i,a]) != 0 and np.linalg.det(Car[:,:,i+1,a]) != 0:

            Wr_r[:,a] = so3_R3(logm(np.linalg.inv(Car[:,:,i,a])@Car[:,:,i+1,a]))/dt
        else:
            Wr_r[:,a] = np.zeros(3)
        #yaw = R.from_matrix(Car[:,:,i,a]).as_euler('zyx')[0]
    # ic(X,yaw)
    # input()

# Initialisation de la figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Affichage des axes d'orientation
scale = 0.5  # Taille des axes
for i in range(N):
    # Matrice de rotation
    origin = ra_r[:,0,i]
    R = Car[:,:,i,0]
    # Vecteurs propres
    x_axis = R[:,0] * scale
    y_axis = R[:,1] * scale
    z_axis = R[:,2] * scale



    # Affichage des vecteurs propres
    # Tracé des axes
    ax.quiver(*origin, *x_axis, color='r', linewidth=1, label="X local" if i == 0 else "")
    ax.quiver(*origin, *y_axis, color='g', linewidth=1, label="Y local" if i == 0 else "")
    ax.quiver(*origin, *z_axis, color='b', linewidth=1, label="Z local" if i == 0 else "")

# Légendes et affichage
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlim(-2, 2)
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