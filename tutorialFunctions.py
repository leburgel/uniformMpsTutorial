import numpy as np
from scipy.linalg import rq
from scipy.linalg import qr
from scipy.linalg import svd
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
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
    # rhoL is eigenvector with 2 legs (bottom - top)

    T = createTransfer(A)
    D = A.shape[0]

    # eigs: k = amount of eigenvalues
    #    which = 'LM' selects largest magnitude eigenvalues
    lam, rhoL = eigs(np.resize(T, (D**2, D**2)).T, k=1, which='LM')

    return lam, np.resize(rhoL, (D, D)).T # hier mogelijks nog transpose nodig !!

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

    return rhoL, rhoR/trace

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
    # L * A = A_L * L
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

def rightOrthonormal(A):
    # function that brings MPS A into right orthonormal gauge, such that
    # A * R = R * A_R
    # returns (R, A_R)

    D = A.shape[0]
    d = A.shape[1]

    # random guess for R
    R = np.random.rand(D,D)
    convergence = 1

    # Decompose A*R until R converges
    while convergence > 1e-8:
        AR = np.einsum('ijk,kl->ijl', A, R)
        Rnew, A_R = RQPositive(np.resize(AR, (D, D*d)))
        convergence = np.linalg.norm(Rnew-R)
        R = Rnew

    return R, np.resize(A_R, (D,d,D))

def entanglementSpectrum(aL, aR, l, r, truncate=0):
    #aL and aR are left and right MPS tensors
    #l and r bring A in left or right form respectively
    #find Schmidt coefficients
    #calculate bipartite entanglement entropy
    #apply truncation if desired

    #center matrix c is matrix multiplication of l and r
    c = l @ r

    #singular value decomposition
    u,s,v = svd(c)

    #for well defined l and r, normalisation probably not necessary but just in case
    s = s / s[0]

    #apply a truncation step keep 'truncate' singular values
    if truncate:
        s = s[:truncate]
        u = u[:,:truncate]
        v = v[:truncate,:]

        #transform aL and aR through unitary
        aLU = np.einsum('ij,jkl->ikl', np.conj(u).T, aL)
        aLU = np.einsum('ikl,lm->ikm', aLU, u)
        aRV = np.einsum('ij,jkl->ikl', v, aR)
        aRV = np.einsum('ikl,lm->ikm', aRV, np.conj(v).T)

        #calculate entropy through singular values
        entropy = -np.sum(s ** 2 * np.log(s ** 2))

        return aLU, aRV, s, entropy

    #transform aL and aR through unitary
    aLU = np.einsum('ij,jkl->ikl', np.conj(u).T, aL)
    aLU = np.einsum('ikl,lm->ikm', aLU, u)
    aRV = np.einsum('ij,jkl->ikl', v, aR)
    aRV = np.einsum('ikl,lm->ikm', aRV, np.conj(v).T)

    # calculate entropy through singular values
    entropy = -np.sum(s**2*np.log(s**2))

    return aLU, aRV, s, entropy

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
    return -Jx*np.einsum('ij,kl->ijkl',Sx, Sx)-Jy*np.einsum('ij,kl->ijkl',Sy, Sy)-Jz*np.einsum('ij,kl->ijkl',Sz, Sz) /
            - h*np.einsum('ij,kl->ijkl',I,Sz) - h*np.einsum('ij,kl->ijkl',Sz,I)
