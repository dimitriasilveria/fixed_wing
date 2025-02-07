from pyfly.pyfly import PyFly
from pyfly.pid_controller import PIDController
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import R3_so3, so3_R3
from icecream import ic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#reference trajectory
N = 50 # number of points
r = 2 # radius
w = 1 # angular velocity
dt = 0.1 # time step
x = np.zeros((N, 3))
x_dot = np.zeros((N, 3))
x_dot_dot = np.zeros((N, 3))
Car = np.zeros((3,3,N))
Wr_r = np.zeros((3,1))
for i in range(N):
    t = i*dt
    x[i] = np.array([r*np.cos(w*t), r*np.sin(w*t), 2])
    x_dot[i] = np.array([-r*w*np.sin(w*t), r*w*np.cos(w*t),0])
    x_dot_dot[i] = np.array([-r*w*w*np.cos(w*t), -r*w*w*np.sin(w*t),0])



fixed_wing = PyFly("/home/bitdrones/fixed_wing/pyfly/pyfly/pyfly_config.json", "/home/bitdrones/fixed_wing/pyfly/pyfly/x8_param.mat")
fixed_wing.seed(0)

fixed_wing.reset(state={"roll": -0.5, "pitch": 0.15})


for i in range(N-1):
    phi = fixed_wing.state["roll"].value
    theta = fixed_wing.state["pitch"].value
    Va = fixed_wing.state["Va"].value
    omega = [fixed_wing.state["omega_p"].value, fixed_wing.state["omega_q"].value, fixed_wing.state["omega_r"].value]
    #simplified_forces(self, attitude, Car, vel, acc)
    X = np.arctan2(x_dot[i,1],x_dot[i,0])

    ca_1 = np.array([np.cos(X),np.sin(X),0]).T #auxiliar vector 
    attitude = R.from_matrix(Car[:,:,i]).as_quat()
    

    fa_r = fixed_wing.simplified_forces(attitude, Car[:,:,i], x_dot[i,:], x_dot_dot[i,:],Wr_r)
    r1 = x_dot[i,:].reshape(3,1)/np.linalg.norm(x_dot[i,:])
    if np.linalg.norm(fa_r) != 0:
        r3 = fa_r.reshape(3,1)/np.linalg.norm(fa_r)
    else:
        r3 = np.zeros((3,1))

    aux = R3_so3(r3)@r1
    if np.linalg.norm(aux) != 0:
        r2 = aux.reshape(3,1)/np.linalg.norm(aux);
    else:
        r2 = np.zeros((3,1))

    Car[:,:,i+1] = np.hstack((r1, r2, r3))

    if np.linalg.det(Car[:,:,i]) != 0 and np.linalg.det(Car[:,:,i+1]) != 0:
        ic(np.linalg.inv(Car[:,:,i]),Car[:,:,i+1])
        Wr_r = so3_R3(np.linalg.inv(Car[:,:,i])@Car[:,:,i+1])/dt
    else:
        Wr_r = np.zeros((3,1))
    yaw = R.from_matrix(Car[:,:,i]).as_euler('zyx')[0]
    # ic(X,yaw)
    # input()

# Initialisation de la figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Affichage des axes d'orientation
scale = 0.5  # Taille des axes
for i in range(N):
    # Matrice de rotation
    origin = x[i,:]
    R = Car[:,:,i]
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
plt.show()