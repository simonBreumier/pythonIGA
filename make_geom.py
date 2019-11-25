import math
import numpy as np

def build_BC_surf(knotvector_u, ctrlpts, idsurf):
	nobU, nobV = ctrlpts.shape[0:2]
	if idsurf == 4:
		BC_ctrlpts = ctrlpts[:,-1,:]
		BC_corres = []
		for i in range(0,nobU):
			BC_corres.append((nobV-1)*nobU+i)
		BC_knot = knotvector_u
		BC_knot = np.array(BC_knot)
	elif idsurf == 1:
		BC_ctrlpts = ctrlpts[:, 0, :]
		BC_corres = []
		for i in range(0, nobU):
			BC_corres.append((nobV - 1) * nobU + i)
		BC_knot = knotvector_u
		BC_knot = np.array(BC_knot)
	return BC_ctrlpts, BC_knot, BC_corres


def make_holeplate_deg3(R, L):
	p = 3
	q = 3

	w = 0.5 * (1 + 1 / math.sqrt(2))
	ctrlpts = np.zeros((6,4, 3))
	ctrlpts[:, 0, :] = [[-R, 0, 1],
						[-0.9024, 0.2357, 0.9024],
						[-0.7702,  0.4369, 0.8536],
						[-0.4369, 0.7702,  0.8536],
						[-0.2357, 0.9024,  0.9024],
						[0., 1., 1.]]

	ctrlpts[:, 1, :] = [[-2., 0, 1],
						[-1.9675, 0.4119, 0.9675],
						[-1.7290, 0.8401, 0.9512],
						[-0.8401, 1.7290, 0.9512],
						[ -0.4119,  1.967, 0.9675],
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
	knotvector_u = [0, 0, 0,0, 0.5, 0.5,1, 1, 1, 1]
	knotvector_v = [0, 0, 0,0,1, 1, 1, 1]

	return p, q, ctrlpts, knotvector_u, knotvector_v

def make_holeplate(R, L):
	p = 2
	q = 2

	w = 0.5*(1+1/math.sqrt(2))
	ctrlpts = np.zeros((4,3,3))
	fact = math.pi/6
	ctrlpts[:,0,:] = [[-R, 0, 1],
					[-R*math.cos(fact), R*math.sin(fact), w],
					[-R*math.cos(2*fact), R*math.sin(2*fact), w],
					[0., 1., 1]]

	ctrlpts[:,1,:] = [[-2.5, 0., 1.],
					[-2.5, 0.75, 1.],
					[-0.75, 2.5, 1.],
					[0., 2.5, 1.]];

	ctrlpts[:,2,:] = [[-L, 0.,1.],
					[-L, L, 1.],
					[-L, L, 1.],
					[0., L, 1.]]
	# Set knot vectors
	knotvector_u = [0,0,0,0.5,1,1,1]
	knotvector_v = [0,0,0,1,1,1]
	
	return p,q,ctrlpts, knotvector_u, knotvector_v

def make_holeplate3(R, L):
	p = 3
	q = 2

	w = 0.5*(1+1/math.sqrt(2))
	ctrlpts = np.zeros((7,3,3))
	ctrlpts[:,0,:] = [[-R, 0, 1],
					[-0.9024, 0.2357, 0.9024],
					[-0.7702, 0.4368, 0.8536],
					[-0.6036, 0.6036, 0.8536],
					[-0.4369, 0.7702, 0.8536],
					[-0.2357, 0.9024, 0.9024],
					[0., 1., 1.]]
	fact =  math.pi/12
	# ctrlpts[:,0,:] = [[-math.cos(0*fact), 0, 1],
	# 				[-math.cos(fact), math.sin(fact), math.cos(fact)],
	# 				[-math.cos(2*fact), math.sin(2*fact), math.cos(2*fact)],
	# 				[-math.cos(3*fact), math.sin(3*fact), math.cos(2*fact)],
	# 				[-math.cos(4*fact), math.sin(4*fact), math.cos(2*fact)],
	# 				[-math.cos(5*fact), math.sin(5*fact), math.cos(fact)],
	# 				[0., 1., 1.]]
					
	ctrlpts[:,1,:] = [[-2.5, 0, 1],
					[-2.4512, 0.7845, 0.9512],
					[-2.3851, 1.5518, 0.9268],
					[-2.3018, 2.3018, 0.9268], 
					[-1.5518, 2.3851, 0.9268], 
					[-0.7845, 2.4512, 0.9512], 
					[0., 2.5, 1.]]
					
	ctrlpts[:,2,:] = [[-4., 0, 1],
					[-4., 1.3333, 1.],
					[-4., 2.6667,1.],
					[-4., 4., 1.], 
					[-2.6667, 4., 1.], 
					[-1.3333, 4., 1.], 
					[0., 4. ,1.]]

	# Set knot vectors
	knotvector_u = [0,0,0,0,0.5,0.5,0.5,1,1,1,1]
	knotvector_v = [0,0,0,1,1,1]
	
	return p,q,ctrlpts, knotvector_u, knotvector_v
	
def make_square(L):
	p = 3
	q = 3

	w = 0.5
	ctrlpts = np.zeros((4,4,3))
	# ctrlpts[:,0,:] = [[0, 0, 1],
					# [0.5*L, 0., 1],
					# [L, 0, 1]]
					
	# ctrlpts[:,1,:] = [[0., 0.5*L, 1],
					# [0.5*L,0.5*L, w],
					# [L,0.5*L,1]]
					
	# ctrlpts[:,2,:] = [[0., L ,1.],
					# [0.5*L, L, w],
					# [L, L ,1.]]
	ctrlpts[:,0,:] = [[0, 0, 1],
					[0.333*L, 0., 1],
					[0.667*L, 0, 1],
					[L, 0, 1]]
					
	ctrlpts[:,1,:] = [[0, 0.333*L, 1],
					[0.333*L, 0.333*L, 1],
					[0.667*L, 0.333*L, 1],
					[L, 0.333*L, 1]]
					
	ctrlpts[:,2,:] = [[0, 0.667*L, 1],
					[0.333*L, 0.667*L, 1],
					[0.667*L, 0.667*L, 1],
					[L, 0.667*L, 1]]
	
	ctrlpts[:,3,:] = [[0, L, 1],
					[0.333*L, L, 1],
					[0.667*L, L, 1],
					[L, L, 1]]

	# Set knot vectors
	knotvector_u = [0,0,0,0,1,1,1,1]
	knotvector_v = [0,0,0,0,1,1,1,1]
	
	return p,q,ctrlpts, knotvector_u, knotvector_v


def make_beam2(L,h):
	p = 2
	q = 2

	ctrlpts = np.zeros((3,3,3))
	ctrlpts[:,0,:] = [[0, 0, 1],
					[0.5*L, 0., 1],
					[L, 0, 1]]
					
	ctrlpts[:,1,:] = [[0., 0.5*h, 1],
					[0.5*L,0.5*h, 1],
					[L,0.5*h,1]]
					
	ctrlpts[:,2,:] = [[0., h ,1.],
					[0.5*L, h, 1],
					[L, h,1.]]

	# Set knot vectors
	knotvector_u = [0,0,0,1,1,1]
	knotvector_v = [0,0,0,1,1,1]
	
	return p,q,ctrlpts, knotvector_u, knotvector_v
	
def make_beam(L,h):
	p = 3
	q = 3

	w = 0.5
	ctrlpts = np.zeros((4,4,3))
	ctrlpts[:,0,:] = [[0, 0, 1],
					[0.333*L, 0., 1],
					[0.667*L, 0, 1],
					[L, 0, 1]]
					
	ctrlpts[:,1,:] = [[0, 0.333*h, 1],
					[0.333*L, 0.333*h, 1],
					[0.667*L, 0.333*h, 1],
					[L, 0.333*h, 1]]
					
	ctrlpts[:,2,:] = [[0, 0.667*h, 1],
					[0.333*L, 0.667*h, 1],
					[0.667*L, 0.667*h, 1],
					[L, 0.667*h, 1]]
	
	ctrlpts[:,3,:] = [[0, h, 1],
					[0.333*L, h, 1],
					[0.667*L, h, 1],
					[L, h, 1]]

	# Set knot vectors
	knotvector_u = [0,0,0,0,1,1,1,1]
	knotvector_v = [0,0,0,0,1,1,1,1]
	
	return p,q,ctrlpts, knotvector_u, knotvector_v