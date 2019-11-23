from vpython import *
import numpy as np

scene.caption= """A model of a solid represented as atoms connected by interatomic bonds.

To rotate "camera", drag with right button or Ctrl-drag.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
  On a two-button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate."""


def set_to_center_of_mass(R):
    r_center_of_mass = np.mean(R, axis=0)
    for bead in range(N_beads):
        R[bead,:] = R[bead,:] - r_center_of_mass

def return_connector_vectors(R):
    return R[1:] - R[0:-1]

def return_bead_vectors(Q):
    R = np.zeros((N_beads, 3))
    for bead in range(1,N_beads):
        R[bead,:] = R[bead-1,:] + Q[bead-1]
    return R

def get_force_vectors(R):
    Fc = return_connector_vectors(R)
    F = np.zeros((N_beads, 3))
    for bead in range(N_beads):
        if bead==0:
            F[bead,:] = Fc[bead,:]
        elif bead==N_beads-1:
            F[bead,:] = -Fc[bead-1,:]
        else:
            F[bead,:] = Fc[bead,:] - Fc[bead-1,:]
    return F

def get_connector_force_vectors_FENE(Q):
    Ql = np.linalg.norm(Q, axis=1)
    Ql2 = np.square(Ql)
    return np.transpose(np.multiply(np.transpose(Q),1/(1-Ql2/b)))

def get_force_vectors_FENE(R):
    #Qs = return_connector_vectors(R)
    Fc = get_connector_force_vectors_FENE(Q)
    F = np.zeros((N_beads, 3))
    for bead in range(N_beads):
        if bead==0:
            F[bead,:] = Fc[bead,:]
        elif bead==N_beads-1:
            F[bead,:] = -Fc[bead-1,:]
        else:
            F[bead,:] = Fc[bead,:] - Fc[bead-1,:]
    return F


def step_Euler(R):
    #Very simple Euler integration of R    
    F = get_force_vectors(R)
    dW = np.random.normal(0,dt**0.5,(N_beads,3))
    flow_term = np.transpose(np.dot(k,np.transpose(R)))
    R = R + (flow_term + 0.25*F) * dt + 2**(-0.5)*dW
    set_to_center_of_mass(R)
    return R

def step_FENE_semi_implicit(R, k, b, dt):
    # Semi implicit predictor-corrector integration of
    # a FENE spring
    #F = get_force_vectors_FENE(R)
    Q = return_connector_vectors(R)
    dW = np.random.normal(0,dt**0.5,(N_beads-1,3))
    Fc = get_connector_force_vectors_FENE(Q)
    flow_term = np.transpose(np.dot(k,np.transpose(Q)))
    # predicted change in R
    prev_forces = np.insert(Fc[0:-1],0,0,axis=0)
    next_forces = np.insert(Fc[1:],len(Fc)-1,0,axis=0)
    Q_pred = Q + (0.25*(prev_forces - 2*Fc + next_forces) + \
                flow_term)*dt + 1/np.sqrt(2)*dW
    Q_corr = Q_pred
    #R_pred_old = R_pred
    # Corrector steps
    epsilon = 20
    while epsilon > tol:
        flow_term_pred = np.transpose(np.dot(k,np.transpose(Q_pred)))
        for i in range(len(Q)):
            gamma = Q[i,:] + (0.25*(flow_term[i,:] + flow_term_pred[i,:]) + \
                    0.25*(prev_forces[i,:] + next_forces[i,:]))*dt + 1/np.sqrt(2)*dW[i,:]
            norm_gamma = np.linalg.norm(gamma)
            gamma_direction = gamma/norm_gamma
            coeff = [1, -norm_gamma, -b*(1+0.25*dt), b*norm_gamma]
            roots = np.roots(coeff)
            for root in roots:
                if (root>0)&(root<b):
                    Ql = root
            Q_corr[i,:] = gamma_direction*Ql
            Ql2 = Ql**2
            if i!=len(Q)-1:
                prev_forces[i+1,:] = Q_corr[i,:]*(1/(1-Ql2/b))
            if i!=0:
                next_forces[i-1,:] = Q_corr[i,:]*(1/(1-Ql2/b))
        print("Q_pred")
        print(Q_pred)
        print("Q_corr")
        print(Q_corr)
        epsilon = np.sum((Q_pred-Q_corr)**2)
        Q_pred = Q_corr
        print(epsilon)
    R = return_bead_vectors(Q_corr)
    set_to_center_of_mass(R)
    return R

b = 100
N_beads = 5
dt = 0.05
tol = 0.001

#flow tensor
gamma_dot = 10
k = np.array([[0, gamma_dot, 0], [0, 0, 0], [0, 0, 0]])

shear_flow_pointer = arrow(pos=vector(0,0,0), axis = vector(5, 0, 0), \
                           shaftwidth = 0.3, color = color.red)

# bead vectors

R = np.zeros((N_beads, 3))

for bead in range(1,N_beads):
    links = np.random.normal(0,1,3)
    while ((b != 0) & (np.linalg.norm(links) > np.sqrt(b))):
        links = np.random.normal(0,1,3)
    R[bead,:] = R[bead-1,:] + links
    

set_to_center_of_mass(R)

Q = return_connector_vectors(R)

Q_vectors = []
R_vectors = []
beads = []
rods = []

for connector in range(N_beads-1):
    Q_vectors.append(vector(Q[connector,0],Q[connector,1],Q[connector,2]))

for bead in range(N_beads):
    R_vectors.append(vector(R[bead,0],R[bead,1],R[bead,2]))
    beads.append(sphere(pos=R_vectors[bead], radius=0.5))
    if bead != N_beads-1:
        rods.append(cylinder(pos=R_vectors[bead], axis=Q_vectors[bead], radius=0.1))

while True:
    rate(1)
    #R = step_Euler(R)
    R = step_FENE_semi_implicit(R, k, b, dt)
    Q = return_connector_vectors(R)
    
    for connector in range(N_beads-1):
        Q_vectors[connector] = vector(Q[connector,0],Q[connector,1],Q[connector,2])

    for bead in range(N_beads):
        R_vectors[bead] = vector(R[bead,0],R[bead,1],R[bead,2])
        beads[bead].pos = R_vectors[bead]
        if bead != N_beads-1:
            rods[bead].pos = R_vectors[bead]
            rods[bead].axis = Q_vectors[bead]





