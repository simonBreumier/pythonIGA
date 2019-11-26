"""
Interfacing function with the python NURBS library
https://nurbs-python.readthedocs.io/en/latest/
"""
import numpy as np
from geomdl import NURBS
from geomdl import helpers
from geomdl import operations
from geomdl.visualization import VisMPL


def plot_BC(BC_ctrlpts, BC_knot,p):
	"""Plot a boundary curve

	:param BC_ctrlpts: boundary control points
	:param BC_knot: boundary knot vector
	:param p: boundary degree
	"""
	cu = NURBS.Curve()
	cu.degree = p
	cu.ctrlpts = BC_ctrlpts.tolist()
	cu.knotvector = BC_knot
	cu.delta = 0.01
	# Plot the control point polygon and the evaluated curve
	cu.vis = VisMPL.VisCurve2D()
	cu.render()

def get_nurbs_char(surf):
	"""Extract NURBS characteristics from a python NURBS surface

	:param surf: python NURBS surface object
	"""
	p = surf.degree_u
	q = surf.degree_v
	knot_u = surf.knotvector_u
	knot_v = surf.knotvector_v
	ctrlpts = np.array(surf.ctrlpts2d)
	
	return p, q, knot_u, knot_v, ctrlpts
	
def init_nurbs(p, q, knot_u, knot_v, ctrlpts):
	"""Create a python NURBS surface object

	:param p,q: degree in U and V directions
	:param knot_u,knot_v: knot vectors in U and V directions
	:param ctrlpts: control point matrix
	"""
	surf = NURBS.Surface()
	surf.degree_u = p
	surf.degree_v = q
	surf.ctrlpts2d = ctrlpts.tolist()
	surf.knotvector_u = knot_u
	surf.knotvector_v = knot_v
	
	return surf

def degree_elevation(degree, ctrlpts):
	"""Elevate a curve degree using python NURBS helpers.degree_elevation

	:param degree: New degree
	:param ctrlpts: control point matrix
	:return:
	"""
	return helpers.degree_elevation(degree, ctrlpts)
	
def refine_knots(p, q, knot_u, knot_v, ctrlpts, n):
	"""Refine a surface by knot insertion using operations.refine_knotvector

	:param p,q: degree in U and V directions
	:param knot_u,knot_v: knot vectors in U and V directions
	:param ctrlpts: control point matrix
	:param n: refinement density number
	"""
	surf = init_nurbs(p, q, knot_u, knot_v, ctrlpts)
	surf = operations.refine_knotvector(surf, [n,n])
	return get_nurbs_char(surf)


