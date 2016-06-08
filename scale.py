import numpy as np
import random

def scaleNet(A, dn, dm):
	'''
	M: nxm matrix to scale
	dm: change of m
	dn: change of n
	creates two transformation matrices to scale A
	T1: Nxn
	T2: mxM
	NxM = Nxn * nxm * mxM
	return np.array of of supercells of size NxM

	>>> A = np.array([[1]])
	>>> scaleNet(A,2,2)
	array([[1, 1],
	       [1, 1]])

	'''
	n,m = A.shape;
	N,M = (dn*n,dm*m);
	#generates the transformation matrices
	T1 = np.array([[1 if j in range(i*dn,(i+1)*dn) else 0 for j in range (N)] for i in range(int(n))]).T
	T2 = np.array([[1 if j in range(i*dm,(i+1)*dm) else 0 for j in range (M)] for i in range(int(m))])
	return T1.dot(A/(dn*dm)).dot(T2)


def boundedSuperCell(msum,nrows,ncols,b=(float("-inf"),float("inf"))):
	'''
	msum: (float) the sum of the resulting matrix
	nrows: (int) num rows
	ncols: (int) num cols
	b: (tuple, int) the upper and lower bound for the value of each cell in the matrix
	return: a np.array of shape (r,c) with the value of all cells summing to sum, where no cells are equal
	'''
	#cells = np.array([msum- for i in range(ncols*nrows-1)]+[msum])
	cells = []
	size = (ncols*nrows)
	mean = msum / size
	for i in range(ncols*nrows-1):
		cell = random.uniform(b[0],[1])
		msum -= cell
		while cell in cells:
			cell = random.uniform()
		cells.append(cell)



	


