import numpy as np
from scipy.linalg import null_space
from scipy.sparse.linalg import eigs, LinearOperator, gmres
from ncon import ncon
from time import time
from chapter1 import createMPS, normaliseMPS
from chapter2 import Heisenberg, vumps, reducedHamMixed, LhMixed, RhMixed



# Recovering the ground state of spin-1 Heisenberg using vumps from
# previous chapter

# coupling strengths
Jx, Jy, Jz, hz = -1, -1, -1, 0 # Heisenberg antiferromagnet
# Heisenberg Hamiltonian
h = Heisenberg(Jx, Jy, Jz, hz);

# initialize bond dimension, physical dimension
D = 12
d = 3

# initialize random MPS
A = createMPS(D, d);
A = normaliseMPS(A);


# energy optimization using VUMPS
print('Energy optimization using VUMPS:\n')
t0 = time()
E, Al, Ac, Ar, C = vumps(h, D, A0=A, tol=1e-4)
print('\nTime until convergence:', time()-t0, 's\n')
print('Computed energy:', E, '\n')


def Quasiparticle(h, Al, Ar, Ac, C, p, num):

    tol, D, d = 1e-12, Al.shape[0], Al.shape[1]
    # renormalize hamiltonian and find left and right environments
    hTilde = reducedHamMixed(h, Ac, Ar)
    Lh = LhMixed(hTilde, Al, C, tol)
    Rh = RhMixed(hTilde, Ar, C, tol)
    
    def ApplyHeff(x):
        
        x = np.reshape(x, (D*(d-1), D))
        B = ncon((VL, x), ([-1, -2, 1], [1, -3]))
        
        def ApplyELR(x, p):
            x = x.reshape((D,D))
            overlap = ncon((np.conj(C), x),([1, 2], [1, 2]))
            y = ncon((Al, np.conj(Ar), x), ([-1, 3, 1], [-2, 3, 2], [1, 2]))
            y = x - np.exp(1j*p) * (y - overlap * C)
            y = y.reshape(-1)
            return y

        def ApplyERL(x, p):
            x = x.reshape((D,D))
            overlap=ncon((np.conj(C), x), ([1, 2], [1, 2]))
            y = ncon((x, Ar, np.conj(Al)), ([1, 2], [2, 3, -2], [1, 3, -1]))
            y = x - np.exp(1j*p) * (y - overlap * C)
            y = y.reshape(-1)
            return y

        
        # right disconnected
        right = ncon((B, np.conj(Ar)), ([-1, 2, 1], [-2, 2, 1]))
        handleApplyELR = LinearOperator((D**2, D**2), matvec=lambda v: ApplyELR(v,p))
        right = gmres(handleApplyELR, right.reshape(-1), tol=tol)[0]
        right = right.reshape((D,D))
        
        # left disconnected
        left = \
            1*ncon((Lh, B, np.conj(Al)), ([1,2], [2,3,-2],[1,3,-1]))+\
            1*ncon((Al, B, np.conj(Al), np.conj(Al), hTilde), ([1,2,4],[4,5,-2],[1,3,6],[6,7,-1],[3,7,2,5]))+\
            np.exp(-1j*p)*ncon((B, Ar, np.conj(Al), np.conj(Al), hTilde), ([1,2,4],[4,5,-2],[1,3,6],[6,7,-1],[3,7,2,5]))
        handleApplyERL = LinearOperator((D**2, D**2), matvec=lambda v: ApplyERL(v, -p))
        left = gmres(handleApplyERL, left.reshape(-1), tol=tol)[0]
        left = left.reshape((D,D))
        
        y = \
            1*ncon((B,Ar,np.conj(Ar),hTilde),([-1,2,1],[1,3,4],[-3,5,4],[-2,5,2,3]))+\
            np.exp(1j*p)*ncon((Al,B,np.conj(Ar),hTilde),([-1,2,1],[1,3,4],[-3,5,4],[-2,5,2,3]))+\
            np.exp(-1j*p)*ncon((B,Ar,np.conj(Al),hTilde),([4,3,1],[1,2,-3],[4,5,-1],[5,-2,3,2]))+\
            1*ncon((Al,B,np.conj(Al),hTilde),([4,3,1],[1,2,-3],[4,5,-1],[5,-2,3,2]))+\
            np.exp(1j*p)*ncon((Al,Al,np.conj(Al),right,hTilde),([1,2,4],[4,5,6],[1,3,-1],[6,-3],[3,-2,2,5]))+\
            np.exp(2*1j*p)*ncon((Al,Al,np.conj(Ar),right,hTilde),([-1,6,5],[5,3,2],[-3,4,1],[2,1],[-2,4,6,3]))+\
            1*ncon((Lh,B),([-1,1],[1,-2,-3]))+\
            1*ncon((B,Rh),([-1,-2,1],[1,-3]))+\
            np.exp(-1j*p)*ncon((left,Ar),([-1,1],[1,-2,-3]))+\
            np.exp(+1j*p)*ncon((Lh,Al,right),([-1,1],[1,-2,2],[2,-3]))
            
        y = ncon((y, np.conj(VL)), ([1, 2, -2], [1, 2, -1]))
        y = y.reshape(-1)
        
        return y
    
    # find reduced parametrization
    L = np.reshape(np.moveaxis(np.conj(Al), -1, 0), (D, D*d))
    VL = np.reshape(null_space(L), (D, d, D*(d-1)))
    handleHeff = LinearOperator((D**2*(d-1), D**2*(d-1)), matvec=lambda x: ApplyHeff(x))
    e, x = eigs(handleHeff, k=num, which='SR')
    
    return x, e
    
# 3.2. Computing the Haldane gap
p = np.pi
num = 3;
x, e = Quasiparticle(h, Al, Ar, Ac, C, p, num)
print('First triplet: {}'.format(e))
