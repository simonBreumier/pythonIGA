"""
Different functions to generates nurbs from a description or from a file
"""
import math
import numpy as np
import json


def load_geom(geom_name):
    """ Import a NURBS geometry defined by a JSON object with following parameters:

    :param knotVectorU/V: knot vector list in U and V direction
    :param p,q: NURBS order in U and V directions
    :param controlPoints: control point (x,y) coordinate list sorted by U
    :param weights: NURBS weight associated to each control point
    """

    with open(geom_name, "r") as f:
        mesh_dat = json.load(f)

    knot_u = mesh_dat['knotVectorU']
    knot_v = mesh_dat['knotVectorV']
    p = mesh_dat['pU']
    q = mesh_dat['pV']
    nob_u = len(knot_u) - (p + 1)
    nob_v = len(knot_v) - (q + 1)
    ctrl_temp = mesh_dat['controlPoints']

    ctrl_pts = np.zeros((nob_u, nob_v, 3))
    for i in range(0, nob_u):
        for j in range(0, nob_v):
            ctrl_pts[i, j, 0:2] = ctrl_temp[j * nob_u + i][:]

    ctrl_pts[:, :, 2] = np.reshape(mesh_dat['weights'], (nob_u, nob_v))
    return knot_u, knot_v, p, q, nob_u, nob_v, ctrl_pts


def build_BC_surf(knotvector_u, ctrlpts, idsurf):
    """Generate a nurbs curve describing one boundary of a nurbs. Source: geoPDE

    :param knotvector_u: knot vector in u direction
    :param ctrlpts: control points
    :param idsurf: if of the boundary to be generated
    :return: BC_ctrlpts, BC_knot, BC_corres (correspondence table between the boundary nodes and the global nodes)
    """
    nobU, nobV = ctrlpts.shape[0:2]

    if idsurf == 1:
        BC_ctrlpts = ctrlpts[:, 0, :]
    elif idsurf == 2:
        BC_ctrlpts = ctrlpts[0, :, :]
    elif idsurf == 3:
        BC_ctrlpts = ctrlpts[-1, :, :]
    elif idsurf == 4:
        BC_ctrlpts = ctrlpts[:, -1, :]
    else:
        raise Exception("Wrong boundary ID number (only 1 to 4 accepted)")

    BC_corres = []
    for i in range(0, nobU):
        BC_corres.append((nobV - 1) * nobU + i)
    BC_knot = knotvector_u
    BC_knot = np.array(BC_knot)
    return BC_ctrlpts, BC_knot, BC_corres


def make_holeplate_deg3(R, L):
    """Generate a degree 3 plate with whole with two repeated nodes

    :param R: Whole radius
    :param L: Plate length
    :return: p, q, ctrlpts, knotvector_u, knotvector_v
    """
    p = 3
    q = 3

    ctrlpts = np.zeros((6, 4, 3))
    ctrlpts[:, 0, :] = [[-R, 0, 1],
                        [-0.9024, 0.2357, 0.9024],
                        [-0.7702, 0.4369, 0.8536],
                        [-0.4369, 0.7702, 0.8536],
                        [-0.2357, 0.9024, 0.9024],
                        [0., 1., 1.]]

    ctrlpts[:, 1, :] = [[-2., 0, 1],
                        [-1.9675, 0.4119, 0.9675],
                        [-1.7290, 0.8401, 0.9512],
                        [-0.8401, 1.7290, 0.9512],
                        [-0.4119, 1.967, 0.9675],
                        [0., 2., 1.]]

    ctrlpts[:, 2, :] = [[-3., 0, 1],
                        [-3., 1.2222, 1.],
                        [-2.8056, 2.0278, 1.],
                        [-2.0278, 2.8056, 1.],
                        [-1.2222, 3., 1.],
                        [0., 3., 1.]]

    ctrlpts[:, 3, :] = [[-4, 0, 1],
                        [-4, 2.6667, 1.],
                        [-4, 4., 1.],
                        [-4, 4., 1.],
                        [-2.6667, 4., 1.],
                        [0., 4., 1.]]
    # Set knot vectors
    knotvector_u = [0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1]
    knotvector_v = [0, 0, 0, 0, 1, 1, 1, 1]

    return p, q, ctrlpts, knotvector_u, knotvector_v


def make_holeplate(R, L):
    """Generate a degree 2 plate with whole with two repeated nodes

    :param R: Whole radius
    :param L: Plate length
    :return: p, q, ctrlpts, knotvector_u, knotvector_v
    """
    p = 2
    q = 2

    w = 0.5 * (1 + 1 / math.sqrt(2))
    ctrlpts = np.zeros((4, 3, 3))
    fact = math.pi / 6
    ctrlpts[:, 0, :] = [[-R, 0, 1],
                        [-R * math.cos(fact), R * math.sin(fact), w],
                        [-R * math.cos(2 * fact), R * math.sin(2 * fact), w],
                        [0., 1., 1]]

    ctrlpts[:, 1, :] = [[-2.5, 0., 1.],
                        [-2.5, 0.75, 1.],
                        [-0.75, 2.5, 1.],
                        [0., 2.5, 1.]];

    ctrlpts[:, 2, :] = [[-L, 0., 1.],
                        [-L, L, 1.],
                        [-L, L, 1.],
                        [0., L, 1.]]
    # Set knot vectors
    knotvector_u = [0, 0, 0, 0.5, 1, 1, 1]
    knotvector_v = [0, 0, 0, 1, 1, 1]

    return p, q, ctrlpts, knotvector_u, knotvector_v


def make_holeplate3(R, L):
    """Generate a degree 3 (u) and 2 (v) plate with whole without repeated node. Source: geoPDE

        :param R: Whole radius
        :param L: Plate length
        :return: p, q, ctrlpts, knotvector_u, knotvector_v
        """
    p = 3
    q = 2

    ctrlpts = np.zeros((7, 3, 3))
    ctrlpts[:, 0, :] = [[-R, 0, 1],
                        [-0.9024, 0.2357, 0.9024],
                        [-0.7702, 0.4368, 0.8536],
                        [-0.6036, 0.6036, 0.8536],
                        [-0.4369, 0.7702, 0.8536],
                        [-0.2357, 0.9024, 0.9024],
                        [0., 1., 1.]]

    ctrlpts[:, 1, :] = [[-2.5, 0, 1],
                        [-2.4512, 0.7845, 0.9512],
                        [-2.3851, 1.5518, 0.9268],
                        [-2.3018, 2.3018, 0.9268],
                        [-1.5518, 2.3851, 0.9268],
                        [-0.7845, 2.4512, 0.9512],
                        [0., 2.5, 1.]]

    ctrlpts[:, 2, :] = [[-4., 0, 1],
                        [-4., 1.3333, 1.],
                        [-4., 2.6667, 1.],
                        [-4., 4., 1.],
                        [-2.6667, 4., 1.],
                        [-1.3333, 4., 1.],
                        [0., 4., 1.]]

    # Set knot vectors
    knotvector_u = [0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1]
    knotvector_v = [0, 0, 0, 1, 1, 1]

    return p, q, ctrlpts, knotvector_u, knotvector_v


def make_square(L):
    """Generate a degree 3 simple square

        :param L: Square length
        :return: p, q, ctrlpts, knotvector_u, knotvector_v
        """
    p = 3
    q = 3

    ctrlpts = np.zeros((4, 4, 3))
    ctrlpts[:, 0, :] = [[0, 0, 1],
                        [0.333 * L, 0., 1],
                        [0.667 * L, 0, 1],
                        [L, 0, 1]]

    ctrlpts[:, 1, :] = [[0, 0.333 * L, 1],
                        [0.333 * L, 0.333 * L, 1],
                        [0.667 * L, 0.333 * L, 1],
                        [L, 0.333 * L, 1]]

    ctrlpts[:, 2, :] = [[0, 0.667 * L, 1],
                        [0.333 * L, 0.667 * L, 1],
                        [0.667 * L, 0.667 * L, 1],
                        [L, 0.667 * L, 1]]

    ctrlpts[:, 3, :] = [[0, L, 1],
                        [0.333 * L, L, 1],
                        [0.667 * L, L, 1],
                        [L, L, 1]]

    # Set knot vectors
    knotvector_u = [0, 0, 0, 0, 1, 1, 1, 1]
    knotvector_v = [0, 0, 0, 0, 1, 1, 1, 1]

    return p, q, ctrlpts, knotvector_u, knotvector_v


def make_beam2(L, h):
    """Generate a degree 2 beam

        :param h: beam thickness
        :param L: beam length
        :return: p, q, ctrlpts, knotvector_u, knotvector_v
        """
    p = 2
    q = 2

    ctrlpts = np.zeros((3, 3, 3))
    ctrlpts[:, 0, :] = [[0, 0, 1],
                        [0.5 * L, 0., 1],
                        [L, 0, 1]]

    ctrlpts[:, 1, :] = [[0., 0.5 * h, 1],
                        [0.5 * L, 0.5 * h, 1],
                        [L, 0.5 * h, 1]]

    ctrlpts[:, 2, :] = [[0., h, 1.],
                        [0.5 * L, h, 1],
                        [L, h, 1.]]

    # Set knot vectors
    knotvector_u = [0, 0, 0, 1, 1, 1]
    knotvector_v = [0, 0, 0, 1, 1, 1]

    return p, q, ctrlpts, knotvector_u, knotvector_v


def make_beam3(L, h):
    """Generate a degree 3 beam

            :param h: beam thickness
            :param L: beam length
            :return: p, q, ctrlpts, knotvector_u, knotvector_v
            """
    p = 3
    q = 3

    w = 0.5
    ctrlpts = np.zeros((4, 4, 3))
    ctrlpts[:, 0, :] = [[0, 0, 1],
                        [0.333 * L, 0., 1],
                        [0.667 * L, 0, 1],
                        [L, 0, 1]]

    ctrlpts[:, 1, :] = [[0, 0.333 * h, 1],
                        [0.333 * L, 0.333 * h, 1],
                        [0.667 * L, 0.333 * h, 1],
                        [L, 0.333 * h, 1]]

    ctrlpts[:, 2, :] = [[0, 0.667 * h, 1],
                        [0.333 * L, 0.667 * h, 1],
                        [0.667 * L, 0.667 * h, 1],
                        [L, 0.667 * h, 1]]

    ctrlpts[:, 3, :] = [[0, h, 1],
                        [0.333 * L, h, 1],
                        [0.667 * L, h, 1],
                        [L, h, 1]]

    # Set knot vectors
    knotvector_u = [0, 0, 0, 0, 1, 1, 1, 1]
    knotvector_v = [0, 0, 0, 0, 1, 1, 1, 1]

    return p, q, ctrlpts, knotvector_u, knotvector_v
