## Unit tests for Beads.py

import pytest
import Constants as c
import numpy as np
from Beads import *

def test_make_beads_non_positive_size():
    with pytest.raises(AssertionError):
        make_random_beads(0)
    with pytest.raises(AssertionError):
        make_random_beads(-1)

def test_make_beads_shape():
    assert make_random_beads(5).shape == (5, 3)

def test_make_beads_link_length():
    beads = make_random_beads(300, .5)
    for i in range(1, 300):
        assert np.linalg.norm(beads[i] - beads[i-1]) <= .5

def test_move_center_of_mass_to_origin():
    beads = make_random_beads(11)
    move_center_of_mass_to_origin(beads)
    np.testing.assert_almost_equal(np.mean(beads, axis=0)[0], 0)

def test_get_links():
    beads = np.array([(1.0, 1.0, 1.0), (2.0, 0.0, 2.0)])
    np.testing.assert_almost_equal(np.linalg.norm(get_links(beads)), np.sqrt(3))

def test_move_beads_using_links():
    beads = np.array([(1.0, 1.0, 1.0), (2.0, 0.0, 2.0)])
    move_beads_using_links(beads, np.array([(0.0, 1.0, 2.0)]))
    np.testing.assert_array_equal(beads, np.array([(1.0, 1.0, 1.0), (1.0, 2.0, 3.0)]))

def test_get_forces():
    links = np.array([(1.0, 2.0, 0.0)])
    np.testing.assert_array_almost_equal(get_forces(links, 10), np.array([(2.0, 4.0, 0.0)]))
    np.testing.assert_array_almost_equal(get_forces(links, 1), np.array([(-0.25, -0.5, 0.0)]))
