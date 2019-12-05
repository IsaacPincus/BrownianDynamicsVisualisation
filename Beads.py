## Methods for manipulating beads

import numpy as np

def make_random_beads(count, max_link = 0):
    """Make vectors for a set of beads. The first bead will be at the origin. The links between beads will be no longer than max_length"""
    assert count > 0
    beads = np.zeros((count, 3))
    for i in range(1,count):
        links = np.random.normal(0,1,3)
        if (max_link > 0):
            while (np.linalg.norm(links) > max_link):
                links = np.random.normal(0,1,3)
        beads[i,:] = beads[i-1,:] + links
    move_center_of_mass_to_origin(beads)
    return beads

def move_center_of_mass_to_origin(beads):
    """Move the beads to that their center of mass is near the origin"""
    center_of_mass = np.mean(beads, axis=0)
    for bead in range(len(beads)):
        beads[bead,:] = beads[bead,:] - center_of_mass

def get_links(beads):
    """Get the vectors representing the links between successive beads"""
    return beads[1:] - beads[0:-1]

def move_beads_using_links(beads, links):
    """Move the beads to positions represented by the given vector links. The first bead will not be moved"""
    for bead in range(1, len(beads)):
        beads[bead,:] = beads[bead-1,:] + links[bead-1]

def get_forces(links, b):
    """Calculate the forces applying to beads connected with the given links"""
    norm = np.linalg.norm(links, axis=1)
    norm_sq = np.square(norm)
    return np.transpose(np.multiply(np.transpose(links),1/(1-norm_sq/b)))

def step_FENE_semi_implicit(beads, b, tol, dt, shear):
    """Step the simulation forward by a given time step"""
    links_corr = calculate_links_correction(beads, b, tol, dt, shear)
    move_beads_using_links(beads, links_corr)
    move_center_of_mass_to_origin(beads)

def calculate_links_correction(beads, b, tol, dt, shear):
    """Calculate semi implicit predictor-corrector integration of a FENE spring"""
    links = get_links(beads)
    dW = np.random.normal(0,dt**0.5,(len(beads)-1,3))
    forces = get_forces(links, b)
    k = np.array([[0, shear, 0], [0, 0, 0], [0, 0, 0]])
    flow_term = np.transpose(np.dot(k,np.transpose(links)))

    # predicted change in beads
    prev_forces = np.insert(forces[0:-1],0,0,axis=0)
    next_forces = np.insert(forces[1:],len(forces)-1,0,axis=0)
    links_pred = links + (0.25*(prev_forces - 2*forces + next_forces) + \
                flow_term)*dt + 1/np.sqrt(2)*dW
    links_corr = np.copy(links_pred)

    # Corrector steps
    epsilon = 20
    while epsilon > tol:
        flow_term_pred = np.transpose(np.dot(k,np.transpose(links_pred)))
        for i in range(len(links)):
            gamma = links[i,:] + (0.5*(flow_term[i,:] + flow_term_pred[i,:]) + \
                    0.25*(prev_forces[i,:] + next_forces[i,:]))*dt + 1/np.sqrt(2)*dW[i,:]
            norm_gamma = np.linalg.norm(gamma)
            gamma_direction = gamma/norm_gamma
            linksl = get_root(norm_gamma, b, dt)
            links_corr[i,:] = gamma_direction*linksl
            linksl2 = linksl**2
            if i!=len(links)-1:
                prev_forces[i+1,:] = links_corr[i,:]*(1/(1-linksl2/b))
            if i!=0:
                next_forces[i,:] = links_corr[i,:]*(1/(1-linksl2/b))
        epsilon = np.sum((links_pred-links_corr)**2)
        links_pred = np.copy(links_corr)    
    return links_corr

def get_root(norm_gamma, b, dt):
    """Find the first root in range"""
    coeff = [1, -norm_gamma, -b*(1+0.5*dt), b*norm_gamma]
    roots = np.roots(coeff)
    for root in roots:
        if (root>0)&(root<np.sqrt(b)):
            return root
    raise Exception('no root found')
    

