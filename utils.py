import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
import time
import rclpy
import rclpy.logging
D = np.zeros((3,3))
mb= 0.042
g = 9.81
I3 = np.array([0,0,1]).T
w_r = 0 #reference yaw
ca_1 = np.array([np.cos(w_r),np.sin(w_r),0]).T #auxiliar vector 
info = rclpy.logging.get_logger("rclpy").info



def R3_so3(w):
    v3 = w[2,0]
    v2 = w[1,0]
    v1 = w[0,0]
    so3 = np.array([[ 0 , -v3,  v2],
          [v3,   0, -v1],
          [-v2,  v1,   0]])

    return so3

def so3_R3(log_R):


    w1 = -log_R[2,1]
    w2 = log_R[0,2]
    w3 = -log_R[1,0]
    w = np.array([w1,w2,w3]).T
    return w

def SE3_se3_back(SE3):    
    R=SE3[0:3,0:3]
    theta= np.acos((np.trace(R)-1)/2)
    if theta !=0:
        lnR=logm(R) #(theta/(2*sin(theta)))*(R-R')
    else:
        lnR = np.zeros((3,3))

    w= [-lnR[1,2], lnR[0,2], -lnR[0,1]]
    wx=np.array([[0,     -w[2], w[2]],[w[2],   0,   -w[0]],[-w[1], w[0],   0]])
    if(theta==0):
        Vin=np.eye(3)
    else:
        A=np.sin(theta)/theta
        B=(1-np.cos(theta))/(theta**2)
        Vin=np.eye(3)-(1/2)*wx+(1/(theta**2))*(1-(A/(2*B)))*(wx@wx)
    v =Vin@SE3[0:3,3]
    r = Vin@SE3[0:3,4]
    se3=np.array([w.T, v, r])
    return se3

def dX_to_dXi(dC,dv,dr):
    dX = np.array([[dC, dv ,dr],[np.zeros((1,3)), 1, 0], [np.zeros((1,3)), 0, 1]])
    dXi = SE3_se3_back(dX)
    return dXi

def references(fixed_wing, x_dot, x_dot_dot, dt,N,n_agents):
    Car = np.zeros((3,3,N,n_agents))
    Wr_r = np.zeros((3,n_agents))
    fa_r = np.zeros((3,n_agents))
    
    for i in range(N-1):

        for a in range(n_agents):
            X = np.arctan2(x_dot[1,i,a],x_dot[0,i,a])

            ca_1 = np.array([np.cos(X),np.sin(X),0]).T #auxiliar vector 
            attitude = R.from_matrix(Car[:,:,i,a]).as_quat()
            

            fa_r[:,a] = fixed_wing.simplified_forces(attitude, Car[:,:,i,a], x_dot[:,i,a], x_dot_dot[:,i,a],Wr_r[:,a])
            r1 = x_dot[:,i,a].reshape(3,1)/np.linalg.norm(x_dot[:,i,a])
            if np.linalg.norm(fa_r[:,a]) != 0:
                r3 = fa_r[:,a].reshape(3,1)/np.linalg.norm(fa_r[:,a])
            else:
                r3 = np.zeros((3,1))

            aux = R3_so3(r3)@r1
            if np.linalg.norm(aux) != 0:
                r2 = aux.reshape(3,1)/np.linalg.norm(aux)
            else:
                r2 = np.zeros((3,1))

            Car[:,:,i+1,a] = np.hstack((r1, r2, r3))

            if np.linalg.det(Car[:,:,i,a]) != 0 and np.linalg.det(Car[:,:,i+1,a]) != 0:

                Wr_r[:,a] = so3_R3(logm(np.linalg.inv(Car[:,:,i,a])@Car[:,:,i+1,a]))/dt
            else:
                Wr_r[:,a] = np.zeros(3)
    return fa_r,Car, Wr_r