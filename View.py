## Functions for displaying the visual elements of a brownian motion simulation

from vpython import *

scene.caption= """ A bead-FENE-spring-chain integrated using Brownian dynamics
The red arrow demonstrates the direction of shear flow

To rotate "camera", drag with right button or Ctrl-drag.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
  On a two-button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
Touch screen: pinch/extend to zoom, swipe or two-finger rotate. \n \n"""

## pause controls
running = True

def is_running():
    return running

def pause_run(b):
    global running
    running = not running
    if running:
        b.text = "Pause"
    else:
        b.text = "Run"
    
button(text="Pause", pos=scene.title_anchor, bind=pause_run)

## speed controls

def update_speed_caption(s):
    speed_caption.text = '{:1.2f}'.format(s.value)

scene.append_to_caption('Timestep width, lower is slower but more accurate \n')
speed_slider = slider(min=0.01, max=0.1, value=0.05, length=220, bind=update_speed_caption, right=15)
speed_caption = wtext()
update_speed_caption(speed_slider)
scene.append_to_caption('\n')

def get_speed_value():
    return speed_slider.value

## shear controls

def update_shear_caption(s):
    shear_caption.text = '{:1.2f}'.format(s.value)

scene.append_to_caption('Shear rate, affects tumbling rate \n')
shear_slider = slider(min=0.1, max=10, value=1, length=220, bind=update_shear_caption, right=15)
shear_caption = wtext()
update_shear_caption(shear_slider)
scene.append_to_caption('\n')

def get_shear_value():
    return shear_slider.value

shear_flow_pointer = arrow(pos=vector(0,0,0), axis = vector(5, 0, 0), shaftwidth = 0.3, color = color.red)

## simulation visual elements

spheres = []
rods = []

def get_vector(array, i):
    """Get the vpython vector associated with a given element in an array"""
    return vector(array[i, 0], array[i, 1], array[i, 2])

def create_elements(beads, links):
    """Create the visual elements associated with beads and links"""
    assert len(beads) == len(links) + 1
    for i in range(len(beads)):
        spheres.append(sphere(pos=get_vector(beads, i), radius=0.5))
    for i in range(len(links)):
        rods.append(cylinder(pos=get_vector(beads, i), axis=get_vector(links, i), radius=0.1))

def update_elements(beads, links):
    """Update the visual elements associated with beads and links"""
    assert len(beads) == len(links) + 1
    assert len(beads) == len(spheres)
    assert len(links) == len(rods)
    rate(100)
    for i in range(len(beads)):
        spheres[i].pos = get_vector(beads, i)
    for i in range(len(links)):
        rods[i].pos = get_vector(beads, i)
        rods[i].axis = get_vector(links, i)
