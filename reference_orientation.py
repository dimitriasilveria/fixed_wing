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
N =50 # number of points
r = 2 # radius
dt = 0.1 # time step
n_agents = 3
w = 0.1
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
x[:,0, 0] =  1*np.array([r*np.cos(np.deg2rad(220)),r*np.sin(np.deg2rad(220)),0]).T
x[:,0, 1] = 1*np.array([r*np.cos(np.deg2rad(100)),r*np.sin(np.deg2rad(100)),0]).T
x[:,0, 2] = 1*np.array([r*np.cos(np.deg2rad(20)),r*np.sin(np.deg2rad(20)) ,0]).T
x_dot[:,0,0] = 1*np.array([-r*w*np.sin(np.deg2rad(220)),r*w*np.cos(np.deg2rad(220)),0]).T
x_dot[:,0,1] = 1*np.array([-r*w*np.sin(np.deg2rad(100)),r*w*np.cos(np.deg2rad(100)),0]).T
x_dot[:,0,2] = 1*np.array([-r*w*np.sin(np.deg2rad(20)) ,r*w*np.cos(np.deg2rad(20)) ,0]).T
# for i in range(N):
#     t = i*dt
#     x[i] = np.array([r*np.cos(w*t), r*np.sin(w*t), 2])
#     x_dot[i] = np.array([-r*w*np.sin(w*t), r*w*np.cos(w*t),0])
#     x_dot_dot[i] = np.array([-r*w*w*np.cos(w*t), -r*w*w*np.sin(w*t),0])

embedding = Embedding(r, w,k_phi, 'circle',n_agents,x[:,0],dt)

fixed_wing = PyFly("/home/dimitria/fixed_wing/pyfly/pyfly/pyfly_config.json", "/home/dimitria/fixed_wing/pyfly/pyfly/x8_param.mat")
fixed_wing.seed(0)

fixed_wing.reset(state={"roll": -0.5, "pitch": 0.15})


for i in range(N-1):
    phi = fixed_wing.state["roll"].value
    theta = fixed_wing.state["pitch"].value
    Va = fixed_wing.state["Va"].value
    omega = [fixed_wing.state["omega_p"].value, fixed_wing.state["omega_q"].value, fixed_wing.state["omega_r"].value]
    #simplified_forces(self, attitude, Car, vel, acc)
    phi_new, target_r_new, target_v_new, phi_diff_new, distances_new,debug = embedding.targets(x[:,i],i)
    x[:,i+1,:] = target_r_new#*np.random.uniform(0.999,1.001)
    x_dot[:,i+1,:] = ((x[:,i+1,:] - x[:,i,:])/dt)#*np.random.uniform(0.999,1.001)
    x_dot_dot[:,i+1,:] = ((x_dot[:,i+1,:] - x_dot[:,i,:])/dt)#*np.random.uniform(0.999,1.001)
    distances[:,i] = distances_new
    phi_diff[:,i] = phi_diff_new

    for a in range(n_agents):
        X = np.arctan2(x_dot[1,i,a],x_dot[0,i,a])

        ca_1 = np.array([np.cos(X),np.sin(X),0]).T #auxiliar vector 
        attitude = R.from_matrix(Car[:,:,i,a]).as_quat()
        

        fa_r = fixed_wing.simplified_forces(attitude, Car[:,:,i,a], x_dot[:,i,a], x_dot_dot[:,i,a],Wr_r[:,a])
        r1 = x_dot[:,i,a].reshape(3,1)/np.linalg.norm(x_dot[:,i,a])
        if np.linalg.norm(fa_r) != 0:
            r3 = fa_r.reshape(3,1)/np.linalg.norm(fa_r)
        else:
            r3 = np.zeros((3,1))

        aux = R3_so3(r3)@r1
        if np.linalg.norm(aux) != 0:
            r2 = aux.reshape(3,1)/np.linalg.norm(aux);
        else:
            r2 = np.zeros((3,1))

        Car[:,:,i+1,a] = np.hstack((r1, r2, r3))

        if np.linalg.det(Car[:,:,i,a]) != 0 and np.linalg.det(Car[:,:,i+1,a]) != 0:

            Wr_r[:,a] = so3_R3(logm(np.linalg.inv(Car[:,:,i,a])@Car[:,:,i+1,a]))/dt
        else:
            Wr_r[:,a] = np.zeros(3)
        yaw = R.from_matrix(Car[:,:,i,a]).as_euler('zyx')[0]
    # ic(X,yaw)
    # input()

# Initialisation de la figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Affichage des axes d'orientation
scale = 0.5  # Taille des axes
for i in range(N):
    # Matrice de rotation
    origin = x[:,i,0]
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
ax.set_zlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlim(-3, 3)
ax.legend()

fig = plt.figure()
for i in range(n_diff):
    plt.plot(distances[i,0:-1],label=f"Distance agent {i+1}")
plt.ylabel("Distances (m)")
plt.xlabel("Time (s)")
plt.title("Distances between agents")
plt.legend()
fig = plt.figure()
for i in range(n_diff):
    plt.plot(np.rad2deg(phi_diff[i,0:-1]),label=f"Phase difference agent {i+1}")
plt.title("Phase differences between agents")
plt.ylabel("$\phi$ (degrees)")
plt.xlabel("Time (s)")
plt.legend()


plt.show()