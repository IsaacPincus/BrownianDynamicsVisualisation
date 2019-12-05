# Brownian motion simulation

import numpy as np
import Constants as c
from Beads import *
import View as v

def step_FENE_semi_implicit(beads, k, b, dt):
    # Semi implicit predictor-corrector integration of a FENE spring
    links = get_links(beads)
    dW = np.random.normal(0,dt**0.5,(c.BEAD_COUNT-1,3))
    Fc = get_forces(links, c.B)
    flow_term = np.transpose(np.dot(k,np.transpose(links)))

    # predicted change in beads
    prev_forces = np.insert(Fc[0:-1],0,0,axis=0)
    next_forces = np.insert(Fc[1:],len(Fc)-1,0,axis=0)
    links_pred = links + (0.25*(prev_forces - 2*Fc + next_forces) + \
                flow_term)*dt + 1/np.sqrt(2)*dW
    links_corr = np.copy(links_pred)

    # Corrector steps
    epsilon = 20
    while epsilon > c.TOL:
        flow_term_pred = np.transpose(np.dot(k,np.transpose(links_pred)))
        for i in range(len(links)):
            gamma = links[i,:] + (0.5*(flow_term[i,:] + flow_term_pred[i,:]) + \
                    0.25*(prev_forces[i,:] + next_forces[i,:]))*dt + 1/np.sqrt(2)*dW[i,:]
            norm_gamma = np.linalg.norm(gamma)
            gamma_direction = gamma/norm_gamma
            coeff = [1, -norm_gamma, -b*(1+0.5*dt), b*norm_gamma]
            roots = np.roots(coeff)
            for root in roots:
                if (root>0)&(root<np.sqrt(b)):
                    linksl = root
                    break
            links_corr[i,:] = gamma_direction*linksl
            linksl2 = linksl**2
            if i!=len(links)-1:
                prev_forces[i+1,:] = links_corr[i,:]*(1/(1-linksl2/b))
            if i!=0:
                next_forces[i,:] = links_corr[i,:]*(1/(1-linksl2/b))
        epsilon = np.sum((links_pred-links_corr)**2)
        links_pred = np.copy(links_corr)
    move_beads_using_links(beads, links_corr)
    move_center_of_mass_to_origin(beads)
    return beads


#flow tensor
gamma_dot = v.get_shear_value()
k = np.array([[0, gamma_dot, 0], [0, 0, 0], [0, 0, 0]])


# set up bead vectors
beads = make_random_beads(c.BEAD_COUNT, np.sqrt(c.B))
move_center_of_mass_to_origin(beads)
links = get_links(beads)
v.create_elements(beads, links)

while True:
    if v.is_running():
        gamma_dot = v.get_shear_value()
        k = np.array([[0, gamma_dot, 0], [0, 0, 0], [0, 0, 0]])
        beads = step_FENE_semi_implicit(beads, k, c.B, v.get_speed_value())
        links = get_links(beads)
        v.update_elements(beads, links)

