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
    return beads

def move_center_of_mass_to_origin(beads):
    """Move the beads to that their center of mass is near the origin"""
    center_of_mass = np.mean(beads, axis=0)
    for bead in range(beads.shape[0]):
        beads[bead,:] = beads[bead,:] - center_of_mass

def get_links(beads):
    """Get the vectors representing the links between successive beads"""
    return beads[1:] - beads[0:-1]

def move_beads_using_links(beads, links):
    """Move the beads to positions represented by the given vector links. The first bead will not be moved"""
    for bead in range(1, beads.shape[0]):
        beads[bead,:] = beads[bead-1,:] + links[bead-1]


