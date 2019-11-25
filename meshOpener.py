import json
import numpy as np


def load_geom(geom_name):
	""" Import a NURBS geometry defined by a JSON object with following parameters:
	knotVectorU/V: knot vector list in U and V direction
	p, q: NURBS order in U and V directions
	controlPoints: control point (x,y) coordinate list sorted by U
	weights: NURBS weight associated to each control point """

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
