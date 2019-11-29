from vpython import *
import numpy as np

scene.caption= """ A bead-FENE-spring-chain integrated using Brownian dynamics
The red arrow demonstrates the direction of shear flow

To rotate "camera", drag with right button or Ctrl-drag.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
  On a two-button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate. \n \n"""


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
    Q_corr = np.copy(Q_pred)
    #R_pred_old = R_pred
    # Corrector steps
    epsilon = 20
    while epsilon > tol:
        flow_term_pred = np.transpose(np.dot(k,np.transpose(Q_pred)))
        for i in range(len(Q)):
            gamma = Q[i,:] + (0.5*(flow_term[i,:] + flow_term_pred[i,:]) + \
                    0.25*(prev_forces[i,:] + next_forces[i,:]))*dt + 1/np.sqrt(2)*dW[i,:]
            norm_gamma = np.linalg.norm(gamma)
            gamma_direction = gamma/norm_gamma
            coeff = [1, -norm_gamma, -b*(1+0.5*dt), b*norm_gamma]
            roots = np.roots(coeff)
            for root in roots:
                if (root>0)&(root<np.sqrt(b)):
                    Ql = root
                    break
            Q_corr[i,:] = gamma_direction*Ql
            Ql2 = Ql**2
            if i!=len(Q)-1:
                prev_forces[i+1,:] = Q_corr[i,:]*(1/(1-Ql2/b))
            if i!=0:
                next_forces[i,:] = Q_corr[i,:]*(1/(1-Ql2/b))
        epsilon = np.sum((Q_pred-Q_corr)**2)
        Q_pred = np.copy(Q_corr)
    R = return_bead_vectors(Q_corr)
    set_to_center_of_mass(R)
    return R


## INPUTS!! Don't make N_beads >100 or so unless you really wanna see those frames tick past
b = 50
N_beads = 10
dt = 0.05
tol = 0.0001

## widgets etc
running = True

def Run(b):
    global running
    running = not running
    if running: b.text = "Pause"
    else: b.text = "Run"
    
button(text="Pause", pos=scene.title_anchor, bind=Run)

def setspeed(s):
    wt.text = '{:1.2f}'.format(s.value)
scene.append_to_caption('Timestep width, lower is slower but more accurate \n')
sl = slider(min=0.01, max=0.1, value=0.05, length=220, bind=setspeed, right=15)
wt = wtext(text='{:1.2f}'.format(sl.value))
scene.append_to_caption('\n')

def setspeed_shear(s):
    wt_shear.text = '{:1.2f}'.format(s.value)
scene.append_to_caption('Shear rate, affects tumbling rate \n')
shear_slider = slider(min=0.1, max=10, value=1, length=220, bind=setspeed_shear, right=15)
wt_shear = wtext(text='{:1.2f}'.format(shear_slider.value))
scene.append_to_caption('\n')

#flow tensor
gamma_dot = shear_slider.value
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
    rate(100)
    #R = step_Euler(R)

    if running:
        gamma_dot = shear_slider.value
        k = np.array([[0, gamma_dot, 0], [0, 0, 0], [0, 0, 0]])
        R = step_FENE_semi_implicit(R, k, b, sl.value)
        Q = return_connector_vectors(R)
        
        for connector in range(N_beads-1):
            Q_vectors[connector] = vector(Q[connector,0],Q[connector,1],Q[connector,2])

        for bead in range(N_beads):
            R_vectors[bead] = vector(R[bead,0],R[bead,1],R[bead,2])
            beads[bead].pos = R_vectors[bead]
            if bead != N_beads-1:
                rods[bead].pos = R_vectors[bead]
                rods[bead].axis = Q_vectors[bead]

