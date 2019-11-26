"""
Generate mesh integration rules
"""
import numpy as np
import math

def Gauss_rule(nquad):
	"""Generate gauss points coordinates and weights in the reference element

	:param nquad: number of gauss points (2 or 4 implemented)
	:return:
	"""
	if nquad == 2:
		gw = [1.,1.]
		gp = [-0.57735,0.57735]
	elif nquad == 4:
		gw = [0.652145, 0.652145, 0.347855, 0.347855]
		gp = [-0.339981, 0.339981, -0.861136, 0.861136]
	else: 
		raise Exception('Only 2 or 4 Gauss IP are yet supported')
	return gp, gw