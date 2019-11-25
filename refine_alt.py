


def nurb2proj(nob, controlPoints, weights):
	projcoord = controlPoints
	for i in range(0,nob):
		projcoord[i,:] = projcoord[i,:]*weights[i]
	projcoord = [projcoord, weights]
	return projcoord

def FindSpan(n,p,u,U):
	if u == U(n+2):
		knotSpanIndex= n
		return
	low  = p
	high = n+1
	mid  = math.floor((low + high)/2)
	while u <U(mid+1) or u >= U(mid+2) :
		if u < U(mid+1):
			high = mid
		else:
			low = mid
		mid = math.floor((low+high)/2);
	knotSpanIndex = mid;
	return knotSpanIndex

def RefineKnotVectCurve(n,p,U,Pw,X,r):
	dim  = len(Pw)
	Qw   = np.zeros((n+r+2,dim))
	Ubar = np.zeros((1,n+p+1+r))

	m = n+p+1
	a = FindSpan(n,p,X(1),U)
	b = FindSpan(n,p,X(r+1),U)
	b = b+1
	for j in range(0,a-p):
		Qw[j,:] = Pw[j,:]

	for j in range(b-1,n):
		Qw[j+r+1,:] = Pw[j,:]

	for j in range(0,a):
		Ubar[j]= U[j]

	for j in range(b+p, m):
		Ubar[j+r+2] = U[j+1]

	i = b+p-1
	k = b+p+r
	for j in range(r, 0, -1):
		while X[j] <= U[i] and i>a:
			Qw[k-p,:] = Pw[i-p,:]
			Ubar[k] = U[i];
			k=k-1;
			i=i-1;

		Qw[k-p,:] = Qw[k-p+1,:];
		for l in range(0,p):
			ind = k-p+l;
			alfa = Ubar[k+l] - X[j];
			if abs(alfa) == 0:
				Qw[ind,:] = Qw[ind+1,:];
			else
				alfa = alfa/(Ubar[k+l] - U[i-p+l);
				Qw[ind,:] = alfa* Qw[ind,:] + [1-alfa]* Qw[ind+1,:];

		Ubar[k] = X[j];
		k=k-1;
	return Ubar,Qw


def refine_alt(knot_u, knot_v, nobU, nobV, ctrlpts):
	uKnotVectorU = np.unique(knot_u)
	uKnotVectorV = np.unique(knot_v)
	newknotsX = uKnotVectorU[0:-1] + 0.5 * np.diff(uKnotVectorU)
	newknotsY = uKnotVectorV[0:-1] + 0.5 * np.diff(uKnotVectorV)

	#h - refinement(NURBS) in x - direction
	nonewkX = len(newknotsX)
	newprojcoord = np.zeros((nobU * nobV + nonewkX * nobV, 3))
	rstart = 0
	wstart = 0
	for j in range(0,nobV):
		rstop = rstart + nobU - 1
		wstop = wstart + nobU - 1 + nonewkX
		locCP = ctrlpts[:,j,0:2]
		locweights = ctrlpts[:,j,3]
		locprojcoord = nurb2proj(nobU, locCP, locweights)
		tempknotVectorX, tempcontrolPoints = RefineKnotVectCurve(nobU - 1, p, knot_u, locprojcoord, newknotsX, nonewkX - 1);


		newprojcoord(wstart: wstop,:)=tempcontrolPoints;
		wstart = wstop + 1;
		rstart = rstop + 1;

	obj.knotVectorU = tempknotVectorX;
	obj.kU = obj.kU + nonewkX;
	[obj.controlPoints, obj.weights] = proj2nurbs(newprojcoord);
	obj.nobU = obj.nobU + nonewkX;
	% % h - refinement(NURBS) in y - direction)
	nonewkY = size(newknotsY, 2);
	newprojcoord = zeros(obj.nobU * obj.nobV + nonewkY * obj.nobV, 3);
	for i = 1:obj.nobU
			  % create
	index
	for reading controlPoints
	rcpindex     = i:obj.nobU: obj.nobU * obj.nobV;
	locCP = obj.controlPoints(rcpindex,:);
	locweights = obj.weights(rcpindex);
	locprojcoord = nurb2proj(obj.nobV, locCP, locweights);
	% refinement
	of
	y
	[tempknotVectorY, tempcontrolPoints] = ...
	RefineKnotVectCurve(obj.nobV - 1, obj.pV, obj.knotVectorV, locprojcoord, newknotsY, nonewkY - 1);
	wcpindex = i:obj.nobU: obj.nobU * (obj.nobV + nonewkY);
	newprojcoord(wcpindex,:) = tempcontrolPoints;
	end
	obj.knotVectorV = tempknotVectorY;
	obj.kV = obj.kV + nonewkY;
	[obj.controlPoints, obj.weights] = proj2nurbs(newprojcoord);
	obj.nobV = obj.nobV + nonewkY;
