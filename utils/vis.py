# visualization functions
import numpy as np
import networkx as nx
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import utils.hyp_functions as hf
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import patches

# collinearity check. if collinear, draw a line and don't attempt curve
def collinear(a,b,c):
    if np.abs(c[0] - b[0]) < .1**4 and np.abs(c[0]-a[0]) < .1**4:
        return True
    elif np.abs(c[0] - b[0]) < .1**4 or np.abs(c[0]-a[0]) < .1**4:
        return False

    slope1 = np.abs((c[1]-b[1])/(c[0]-b[0]))
    slope2 = np.abs((c[1]-a[1])/(c[0]-a[0]))

    if np.abs(slope1 - slope2) < .1**4:
        return True

    return False

# todo: speed this code up
def get_circle_center(a,b,c):
    m = np.zeros([2,2])
    m[0,0] = 2*(c[0]-a[0])
    m[0,1] = 2*(c[1]-a[1])
    m[1,0] = 2*(c[0]-b[0])
    m[1,1] = 2*(c[1]-b[1])

    v = np.zeros([2,1])
    v[0] = c[0]**2 + c[1]**2 - a[0]**2 - a[1]**2
    v[1] = c[0]**2 + c[1]**2 - b[0]**2 - b[1]**2

    return (np.linalg.inv(m)@v).flatten()

# distance for Euclidean coordinates
def euclid_dist(a,b):
    return np.linalg.norm(a-b)

# angles for arc
def get_angles(center, a):
    if abs(a[0] - center[0]) < 0.1**3:
        if a[1] > center[1] : theta = 90
        else: theta = 270
    else:
        theta = np.rad2deg(np.arctan((a[1]-center[1])/(a[0]-center[0])))    
        # quadrant 3:
        if (a[0]-center[0]) < 0 and (a[1]-center[1]) < 0:
            theta += 180
        # quadrant 2
        if (a[0]-center[0]) < 0 and (a[1]-center[1]) >= 0:
            theta -= 180

    # always use non-negative angles
    if theta < 0: theta += 360    
    return theta

# draw hyperbolic line:
def draw_geodesic(a, b, c, ax, verbose=False):
    if verbose:
        print("Geodesic points are ", a, "\n", b, "\n", c, "\n")    

    is_collinear = False
    if collinear(a,b,c):
        is_collinear = True
    else:
        cent = get_circle_center(a,b,c)
        radius = euclid_dist(a, cent)
        t1 = get_angles(cent, b)
        t2 = get_angles(cent, a)

        if verbose:
            print("\ncenter at ", cent)
            print("radius is ", radius)
            print("angles are ", t1, " ", t2)
            print("dist(a,center) = ", euclid_dist(cent,a))
            print("dist(b,center) = ", euclid_dist(cent,b))
            print("dist(c,center) = ", euclid_dist(cent,c))

    # if the angle is really tiny, a line is a good approximation
    if is_collinear or (np.abs(t1-t2) < 2):
        coordsA = "data"
        coordsB = "data"
        e = patches.ConnectionPatch(a, b, coordsA, coordsB, linewidth=2)
    else:
        if verbose:
            print("angles are theta_1 = ", t1, " theta_2 = ", t2)
        if (t2>t1 and t2-t1<180) or (t1>t2 and t1-t2>=180):
            e = patches.Arc((cent[0], cent[1]), 2*radius, 2*radius,
                     theta1=t1, theta2=t2, linewidth=2, fill=False, zorder=2)
        else:
            e = patches.Arc((cent[0], cent[1]), 2*radius, 2*radius,
                     theta1=t2, theta2=t1, linewidth=2, fill=False, zorder=2)
    ax.add_patch(e)
    ax.plot(a[0], a[1], "o")
    ax.plot(b[0], b[1], "o")

# to draw geodesic between a,b, we need
# a third point. easy with inversion
def get_third_point(a,b):
    b0 = hf.reflect_at_zero(a,b)
    c0 = b0/2.0
    c = hf.reflect_at_zero(a,c0)

    return c

# for circle stuff let's just draw the points
def draw_points_on_circle(a, ax):
    ax.plot(a[0], a[1], "o")

# draw the embedding for a graph 
# G is the graph, m is the PyTorch hyperbolic model
def draw_graph(G, m, fig, ax):
    hyperbolic_setup(fig, ax[0])
    spherical_setup(fig, ax[1])

    # todo: directly read edge list from csr format
    Gr = nx.from_scipy_sparse_matrix(G)
    for edge in Gr.edges():
        idx = torch.LongTensor([edge[0], edge[1]])
        a = ((torch.index_select(m.H[0].w, 0, idx[0])).clone()).detach().numpy()[0]
        b = ((torch.index_select(m.H[0].w, 0, idx[1])).clone()).detach().numpy()[0]
        c = get_third_point(a,b)
        draw_geodesic(a,b,c,ax[0])

    for node in Gr.nodes():
        idx = torch.LongTensor([int(node)])
        v = ((torch.index_select(m.S[0].w, 0, idx)).clone()).detach().numpy()[0]
        draw_points_on_circle(v, ax[1])

def setup_plot(draw_circle=False):
    # create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,10))

    #fig = plt.figure(figsize=(10,3))
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)

    #fig.set_size_inches(20.0, 10.0, forward=True)
    ax = (ax1, ax2)
    writer = animation.ImageMagickFileWriter(fps=10, metadata=dict(artist='HazyResearch'), bitrate=1800)
    writer.setup(fig, 'HypDistances.gif', dpi=100)

    if draw_circle:
        hyperbolic_setup(fig, ax[0])
        spherical_setup(fig, ax[1])
    return fig, ax, writer


def hyperbolic_setup(fig, ax):
    #fig.set_size_inches(20.0, 10.0, forward=True)
    
    # set axes
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    # draw Poincare disk boundary
    e = patches.Arc((0,0), 2.0, 2.0,
                     linewidth=2, fill=False, zorder=2)
    ax.add_patch(e)

def spherical_setup(fig, ax):
#    fig.set_size_inches(20.0, 10.0, forward=True)
    
    # set axes
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    # draw circle boundary
    e = patches.Arc((0,0), 2.0, 2.0,
                     linewidth=1, fill=False, zorder=2)
    ax.add_patch(e)

def draw_plot():
    plt.show()

def clear_plot():
    plt.cla()
