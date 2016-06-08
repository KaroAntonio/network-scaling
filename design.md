# Network Scaling

	nxm * mxb = nxb
	2x2 * 2x4 = B
	B^T = 4x2
	4x2 * 2x4 = C
	C^T
	
	1	2  
	3	4  
	*  
	1	1	0	0   
	0	0	1	1  
	 = B =  
	1	1	2	2  
	3	3	4	4  
	
	1	3
	1	3
	2	4
	2	4
	*  
	1	1	0	0   
	0	0	1	1  
	= C = 
	1	1	3	3
	1	1	3	3
	2	2	4	4
	2	2	4	4
	
	C^T = 
	1	1	2	2
	1	1	2	2	
	3	3	4	4
	3	3	4	4
	
	So...
	
	given A, shape(A) = nxm
	to transform to a n'xm' matrix
	B = A * T1, shape(T1) = nxm', T1 is a checkerboard matrix
	C = B^T * T1
	Desired M is C^T
	
	nxm -> NxM
	N = n*dn
	M = m*dm 
	
	Nxn * nxm * mxM
	T1 = Nxn
	T2 = mxM
	
	S(A) =	T1 * A * T2
	
	*if matrix is square T1 = T2^T
	
	with numpy:
	n=2itu
	T1 = np.array([([1]*n)+([0]*n),([0]*n)+([1]*n)])
	A.dot(T1).T.dot(T1).T
	
if all the weights are explicitly doubled