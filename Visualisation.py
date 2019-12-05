# Brownian motion simulation

import numpy as np
import Constants as c
import Beads as b
import View as v

# set up bead vectors
beads = b.make_random_beads(c.BEAD_COUNT, np.sqrt(c.B))
v.create_elements(beads, b.get_links(beads))

while True:
    if v.is_running():
        b.step_FENE_semi_implicit(beads, c.B, c.TOL, v.get_speed_value(), v.get_shear_value())
        v.update_elements(beads, b.get_links(beads))

