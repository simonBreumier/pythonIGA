from geomdl import NURBS
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl import operations
from geomdl import helpers
from geomdl.visualization import VisMPL
import numpy as np
import math

def plot_BC(BC_ctrlpts, BC_knot,p):
	cu = NURBS.Curve()
	cu.degree = p
	cu.ctrlpts = BC_ctrlpts.tolist()
	cu.knotvector = BC_knot
	cu.delta = 0.01
	# Plot the control point polygon and the evaluated curve
	cu.vis = VisMPL.VisCurve2D()
	cu.render()

def get_nurbs_char(surf):
	p = surf.degree_u
	q = surf.degree_v
	knot_u = surf.knotvector_u
	knot_v = surf.knotvector_v
	ctrlpts = np.array(surf.ctrlpts2d)
	
	return p, q, knot_u, knot_v, ctrlpts
	
def init_nurbs(p, q, knot_u, knot_v, ctrlpts):
	surf = NURBS.Surface()
	surf.degree_u = p
	surf.degree_v = q
	surf.ctrlpts2d = ctrlpts.tolist()
	surf.knotvector_u = knot_u
	surf.knotvector_v = knot_v
	
	return surf

def degree_elevation(degree, ctrlpts):
	return helpers.degree_elevation(degree, ctrlpts)
	
def refine_knots(p, q, knot_u, knot_v, ctrlpts, n):
	surf = init_nurbs(p, q, knot_u, knot_v, ctrlpts)
	surf = operations.refine_knotvector(surf, [n,n])
	return get_nurbs_char(surf)


