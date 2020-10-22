import numpy as np
from scipy.linalg import rq
from scipy.linalg import qr
from scipy.linalg import svd
from scipy.sparse.linalg import eigs, LinearOperator, gmres
from scipy.optimize import minimize
from functools import partial

def createMPS(bondDimension, physDimension):
    # function to create a random MPS tensor for some bondDimension and physical dimension.
    # returns a 3-legged tensor (leftLeg - physLeg - rightLeg)

    return np.random.rand(bondDimension, physDimension, bondDimension) \
        + 1j*np.random.rand(bondDimension, physDimension, bondDimension)

def createTransfer(A):
    # function to return a transfer matrix starting from a given MPS tensor.
    # returns a 4-legged tensor (topLeft - bottomLeft - topRight - bottomRight)

    return np.einsum('isk,jsl->ijkl', A, np.conj(A))

def leftFixedPointNaive(A):
    # function to find fixed point of MPS transfer matrix, naive implementation
    # using diagonalisation of D^2 by D^2 matrix
    # returns (lam, rhoL)
    # lam is eigenvalue
    # rhoL is eigenvector with 2 legs (bottom - top)!!!!!

    T = createTransfer(A)
    D = A.shape[0]

    # eigs: k = amount of eigenvalues
    # which = 'LM' selects largest magnitude eigenvalues
    lam, rhoL = eigs(np.resize(T, (D**2, D**2)).T, k=1, which='LM')

    return lam, np.resize(rhoL, (D, D)).T

def rightFixedPointNaive(A):
    # function to find fixed point of MPS transfer matrix, naive implementation
    # using diagonalisation of D^2 by D^2 matrix
    # returns (lam, rhoR)
    # lam is eigenvalue
    # rhoR is eigenvector with 2 legs (top - bottom)

    T = createTransfer(A)
    D = A.shape[0]

    # eigs: k = amount of eigenvalues
    #    which = 'LM' selects largest magnitude eigenvalues
    lam, rhoR = eigs(np.resize(T, (D**2, D**2)), k=1, which='LM')

    return lam, np.resize(rhoR, (D, D))

def normaliseMPS(A):
    # function that normalises a given MPS such that the dominant eigenvalue
    # of its transfer matrix is 1
    # returns (Anorm, lam) such that Anorm = lam*A

    lam, _ = rightFixedPoint(A)
    Anorm = A / np.sqrt(lam)

    return Anorm, lam

def normaliseFixedPoints(rhoL, rhoR):
    # function that normalises given left and right fixed points
    # such that they trace to unity (interpretation as density matrix)
    # returns (rhoL, rhoR)

    trace = np.einsum('ij,ji->', rhoL, rhoR)
    # trace = np.trace(rhoL*rhoR) # might work as well/be faster than einsum?
    norm = np.sqrt(trace)
    return rhoL/norm, rhoR/norm

def matrixSqrt(M):
    # function that square roots the eigenvalues of M (have to be positive!)
    # returns L
    # such that LL = M
    S, U = np.linalg.eig(M)
    S = np.diag(np.sqrt(S))
    return U@S@U.T


def rightHandle(A, v):
    # function that implements the action of a transfer matrix defined by A
    # on a right vector of dimension D**2 v (top - bottom)
    # returns a vector of dimension D**2 (top - bottom)

    D = A.shape[2]

    # contraction sequence: contract A with v, then with Abar
    newV = np.einsum('ijk,kl->ijl', A, v.reshape((D,D)))
    newV = np.einsum('ijk,ljk->il', newV, np.conj(A))

    return np.reshape(newV, D**2)

def rightFixedPoint(A):
    # function to find right fixed point of MPS transfer matrix, using smart algorithm
    # returns (lam, rhoR)
    # lam is eigenvalue
    # rhoR is eigenvector with 2 legs (top - bottom)

    T = createTransfer(A)
    D = A.shape[2]

    # create function handle instead of D**2 matrix
    transferRight = LinearOperator( (D**2,D**2), matvec=partial(rightHandle, A))

    # eigs: k = amount of eigenvalues
    #    which = 'LM' selects largest magnitude eigenvalues
    lam, rhoR = eigs(transferRight, k=1, which='LM')

    return lam, np.resize(rhoR, (D, D))

def leftHandle(A,v):
    # function that implements the action of a transfer matrix defined by A
    # on a left vector of dimension D**2 v (bottom - top)
    # returns a vector of dimension D**2 (bottom - top)

    D = A.shape[0]

    # contraction sequence: contract A with v, then with Abar
    newV = np.einsum('ijk,li->ljk', A, v.reshape((D,D)))
    newV = np.einsum('ljk,ljm->mk', newV, np.conj(A))
    return np.reshape(newV, D**2)

def leftFixedPoint(A):
    # function to find fixed point of MPS transfer matrix, naive implementation
    # using diagonalisation of D^2 by D^2 matrix
    # returns (lam, rhoR)
    # lam is eigenvalue
    # rhoR is eigenvector with 2 legs (top - bottom)

    T = createTransfer(A)
    D = A.shape[2]

    # create function handle instead of D**2 matrix
    transferLeft = LinearOperator( (D**2,D**2), matvec=partial(leftHandle, A))

    # eigs: k = amount of eigenvalues
    #    which = 'LM' selects largest magnitude eigenvalues
    lam, rhoR = eigs(transferLeft, k=1, which='LM')

    return lam, np.resize(rhoR, (D, D))

def QRPositive(A):
    # function that implements a QR decomposition of a matrix A, such
    # that the diagonal elements of R are positive, R is upper triangular and
    # Q is an isometry with A = QR
    # returns (Q, R)
    
    D = A.shape[1]
    # QR decomposition, scipy conventions: Q.shape = (D*d, D*d), R.shape = (D*d, D)
    Q, R = qr(A)
    
    # Throw out zeros under diagonal: Q.shape = (D*d, D), R.shape = (D, D)
    Q = Q[:, :D]
    R = R[:D, :]

    # extract signs and multiply with signs on diagonal
    diagSigns = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, diagSigns)
    R = np.dot(diagSigns, R)
    
    return Q, R

def leftOrthonormal(A, tol=1e-14):
    # function that brings MPS A into left orthonormal gauge, such that
    # L  A = A_L  L
    # returns (L, A_L)
    
    D = A.shape[0]
    d = A.shape[1]
    
    # random guess for L
    L = np.random.rand(D,D)
    L = L / np.linalg.norm(L)
    
    
    # Decompose L*A until L converges (infinite loop if not convergent)
    convergence = 1
    while convergence > tol:
        LA = np.einsum('ik,ksj->isj', L, A)
        A_L, Lnew = QRPositive(np.resize(LA, (D*d, D)))
        Lnew = Lnew / np.linalg.norm(Lnew) # only necessary when working with unnormalised MPS
        convergence = np.linalg.norm(Lnew - L)
        L = Lnew
    
    return L, np.resize(A_L, (D,d,D))


def RQPositive(A):
    # function that implements a RQ decomposition of a matrix A, such that
    # the diagonal elements of R are positive, R is upper triangular and Q
    # is an isometry with A = RQ
    # returns (R, Q)

    D = A.shape[0]
    # RQ decomposition: Q.shape = (D*d, D*d), R.shape = (D, D*d)
    R, Q = rq(A)

    # Throw out zeros under diagonal: Q.shape = (D, D*d), R.shape = (D, D)
    Q = Q[-D:, :]
    R = R[:, -D:]

    # extract signs and multiply with signs on diagonal
    diagSigns = np.diag(np.sign(np.diag(R)))
    Q = np.dot(diagSigns, Q)
    R = np.dot(R, diagSigns)

    return R, Q

def rightOrthonormal(A, tol=1e-14):
    # function that brings MPS A into right orthonormal gauge, such that
    # A * R = R * A_R
    # returns (R, A_R)

    D = A.shape[0]
    d = A.shape[1]

    # random guess for R
    R = np.random.rand(D,D)
    convergence = 1

    # Decompose A*R until R converges
    while convergence > tol:
        AR = np.einsum('ijk,kl->ijl', A, R)
        Rnew, A_R = RQPositive(np.resize(AR, (D, D*d)))
        Rnew = Rnew / np.linalg.norm(Rnew) # only necessary when working with unnormalised MPS
        convergence = np.linalg.norm(Rnew-R)
        R = Rnew

    return R, np.resize(A_R, (D,d,D))

def entanglementSpectrum(aL, aR, L, R, truncate=0):
    #aL and aR are left and right MPS tensors
    #l and r bring A in left or right form respectively
    #find Schmidt coefficients
    #calculate bipartite entanglement entropy
    #apply truncation if desired

    #center matrix c is matrix multiplication of l and r
    C = L @ R

    #singular value decomposition
    U,S,V = svd(C)

    #for well defined l and r, normalisation probably not necessary but just in case
    S = S / S[0]

    #apply a truncation step keep 'truncate' singular values
    if truncate:
        S = S[:truncate]
        U = U[:, :truncate]
        V = V[:truncate, :]

        #transform aL and aR through unitary
        aLU = np.einsum('ij,jkl->ikl', np.conj(U).T, aL)
        aLU = np.einsum('ikl,lm->ikm', aLU, U)
        aRV = np.einsum('ij,jkl->ikl', V, aR)
        aRV = np.einsum('ikl,lm->ikm', aRV, np.conj(V).T)

        #calculate entropy through singular values
        entropy = -np.sum(S ** 2 * np.log(S ** 2))

        return aLU, aRV, S, entropy

    #transform aL and aR through unitary
    aLU = np.einsum('ij,jkl->ikl', np.conj(U).T, aL)
    aLU = np.einsum('ikl,lm->ikm', aLU, U)
    aRV = np.einsum('ij,jkl->ikl', V, aR)
    aRV = np.einsum('ikl,lm->ikm', aRV, np.conj(V).T)

    # calculate entropy through singular values
    entropy = -np.sum(S**2*np.log(S**2))

    return aLU, aRV, S, entropy

def CreateSx():
    out = np.zeros((3,3))
    out[0,1] = 1
    out[1,2] = 1
    return (out+out.T)/np.sqrt(2)

def CreateSy():
    out = np.zeros((3,3), dtype=np.complex64)
    out[0,1] = -1j
    out[1,2] = -1j
    return (out+np.conj(out.T))/np.sqrt(2)

def CreateSz():
    out = np.zeros((3,3))
    out[0,0] = 1
    out[2,2] = -1
    return out

def Heisenberg(Jx,Jy,Jz,h):
    Sx, Sy, Sz = CreateSx(),CreateSy(),CreateSz()
    I = np.identity(3)
    return -Jx*np.einsum('ij,kl->ijkl',Sx, Sx)-Jy*np.einsum('ij,kl->ijkl',Sy, Sy)-Jz*np.einsum('ij,kl->ijkl',Sz, Sz) \
            - h*np.einsum('ij,kl->ijkl',I,Sz) - h*np.einsum('ij,kl->ijkl',Sz,I)
            
def oneSiteUniform(O, A, l, r):
    #determine expectation value of one-body operator in uniform gauge
    #first right contraction
    return np.einsum('ijk,mnl,mi,kl,jn', A, np.conj(A), l, r, O)

def oneSiteMixed(O, Ac):
    #determine expectation value of one-body operator in mixed gauge
    #first right contraction
    # ikjl is juiste sequence

    return np.einsum('ijk,jl,ilk', Ac, O, np.conj(Ac))

def twoSiteUniform(H, A, l, r):
    #calculate the expectation value of the hamiltonian H (top left - top right - bottom left - bottom right)
    #that acts on two sites
    #contraction done from right to left
    return np.einsum('ijk,klm,jlqo,rqp,pon,ri,mn', A, A, H, np.conj(A), np.conj(A), l, r)

def twoSiteMixed(H, Ac, Ar):
    #calculate the expectation value of the hamiltonian H (top left - top right - bottom left - bottom right)
    #in mixed canonical form that acts on two sites, contraction done from right to left
    #case where Ac on left legs of H
    # kjlipmno
    return np.einsum('ijk,klm,jlpn,ipo,onm', Ac, Ar, H, np.conj(Ac), np.conj(Ar))

def leftHandle_(A, r, l, v):
    # function that implements the action of 1-T + outer(r,l)
    # on a left vector of dimension D**2 v (bottom - top)
    # returns a vector of dimension D**2 (bottom - top)

    D = r.shape[0]
    v_T = leftHandle(A, v)
    v_rl = np.einsum('ji,ij,kl-> kl', v.reshape((D, D)), r, l)
    return v - v_T + np.reshape(v_rl, D**2)


def rightHandle_(A, r, l, v):
    # function that implements the action of 1-T + outer(r,l)
    # on a left vector of dimension D**2 v (bottom - top)
    # returns a vector of dimension D**2 (bottom - top)

    D = r.shape[0]
    v_T = rightHandle(A,v)
    v_rl = np.einsum('kl,ji,ij-> kl', r, l, v.reshape((D,D)))
    return v - v_T + np.reshape(v_rl, D**2)


def Gradient(H, A, l, r):
    # a rank 3 tensor, equation (116) in the notes
    # consists of 4 terms
    # have to solve x = y(1-T_) where T_ = createTransfer(A) - np.outer(leftFixedPoint(A), rightFixedPoint(A))
    # don't naively construct (1-T_) because all these objects have 4D legs. As before describe how this operator works on a vector y
    # create function handle instead of D**2 matrix
    D = A.shape[0]

    transfer_Left = LinearOperator((D**2, D**2), matvec=partial(leftHandle_, A, r, l))
    x = np.einsum('ijk,klm,jlqo,rqp,pon,ri->mn', A, A, H, np.conj(A), np.conj(A), l)
    x = np.reshape(x, D**2)
    Lh = gmres(transfer_Left, x)[0]

    transfer_Right = LinearOperator((D**2,D**2), matvec=partial(rightHandle_, A, r, l))
    x = np.einsum('ijk,klm,jlqo,rqp,pon,mn->ri', A, A, H, np.conj(A), np.conj(A), r)
    x = np.reshape(x, D**2)
    Rh = gmres(transfer_Right, x.reshape(D**2))[0]

    Lh = np.reshape(Lh, (D, D))
    Rh = np.reshape(Rh, (D, D))
    ###########
    #FIRST TERM
    ###########
    first = np.einsum('ijk,klm,jlqo,pon,ri,mn->rqp', A, A, H, np.conj(A), l, r)
    ###########
    #SECOND TERM
    ###########
    second = np.einsum('ijk,klm,jlqo,rqp,ri,mn->pon', A, A, H, np.conj(A), l, r)

    ###########
    #THIRD TERM
    ###########
    third = np.einsum('mi,ijk,kl->mjl', l, A, Rh)

    ###########
    #FOURTH TERM
    ###########
    fourth = np.einsum('mi,ijk,kl->mjl', Lh, A, r)

    # define Lh and Rh
    return first+second+third+fourth
    
def energyDensity(A, H):
    d = A.shape[1]
    A, _ = normaliseMPS(A)
    _, l = leftFixedPoint(A)
    _, r = rightFixedPoint(A)
    e = np.real(twoSiteUniform(H, A, l, r))
    eOp = e * np.einsum("ik,jl->ijkl", np.identity(d), np.identity(d))
    Htilde = H - eOp
    g = Gradient(Htilde, A, l, r)
    return e, g

def energyWrapper(H, D, d, varA):
    Areal = (varA[:D**2 *d]).reshape(D, d, D)
    Acomplex = (varA[D**2*d:]).reshape(D, d, D)
    A = Areal + 1j*Acomplex
    e, _ = energyDensity(A, H)
    return e

#hier wel nog wat werk denk ik
def gradientWrapper(H, D, d, varA):
    _, g = energyDensity(A, H)
    #extra haakjes om real(g) en imag(g) in tuple te plaatsen voor concate anders error !!!
    g = np.concatenate((np.real(g).reshape(-1), np.imag(g).reshape(-1)))
    return g

### A first test case for the gradient in python
D = 4
d = 3

H = Heisenberg(-1,-1,-1,0)

A = createMPS(D,d)
ReA = np.real(A)
ImA = np.imag(A)

# extra haakjes om real(g) en imag(g) in tuple te plaatsen voor concate anders error !!!
varA = np.concatenate((ReA.reshape(-1), ImA.reshape(-1)))

EnergyHandle = partial(energyWrapper, H, D, d)
gradientHandle = partial(gradientWrapper, H, D, d)

res = minimize(EnergyHandle, varA, jac=gradientHandle)
Aopt = res.x
print(res.fun)