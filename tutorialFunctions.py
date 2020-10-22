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


def normaliseMPS(A):
    """
    Function to normalise a given MPS tensor using O(D^3) algorithm
    input: A --- (D, d, D) MPStensor
    output: A --- (D, d, D) MPStensor
    """

    D = A.shape[0]

    # set optimal contraction sequence
    path = ['einsum_path', (0, 2), (0, 1)]

    # calculate transfer matrix handle and cast to LinearOperator
    transferRightHandle = lambda v: np.reshape(np.einsum('ijk,ljm,km->il', A, np.conj(A), v.reshape((D, D)), optimize=path), D ** 2)
    transferRight = LinearOperator((D ** 2, D ** 2), matvec=transferRightHandle)

    # calculate eigenvalue
    lam = eigs(transferRight, k=1, which='LM', return_eigenvectors=False)

    return A / np.sqrt(lam)


def leftFixedPoint(A):
    """
    Function to determine the left fixed point of a given MPS tensor using O(D^3) algorithm
    input: A --- (D, d, D) MPStensor
    output: l --- (D, D) leftFixedPointTensor (bottom-top)
    """

    D = A.shape[0]

    # set optimal contraction sequence
    path = ['einsum_path', (0, 2), (0, 1)]

    # calculate transfer matrix handle and cast to LinearOperator
    transferLeftHandle = lambda v: np.reshape(
        np.einsum('ijk,ljm,li->mk', A, np.conj(A), v.reshape((D, D)), optimize=path), D ** 2)
    transferLeft = LinearOperator((D ** 2, D ** 2), matvec=transferLeftHandle)

    # calculate fixed point
    _, l = eigs(transferLeft, k=1, which='LM')

    return l.reshape(D, D)


def rightFixedPoint(A):
    """
    Function to determine the right fixed point of a given MPS tensor using O(D^3) algorithm
    input: A --- (D, d, D) MPStensor
    output: r --- (D, D) rightFixedPointTensor (top-bottom)
    """

    D = A.shape[0]

    # set optimal contraction sequence
    path = ['einsum_path', (0, 2), (0, 1)]

    # calculate transfer matrix handle and cast to LinearOperator
    transferRightHandle = lambda v: np.reshape(
        np.einsum('ijk,ljm,km->il', A, np.conj(A), v.reshape((D, D)), optimize=path), D ** 2)
    transferRight = LinearOperator((D ** 2, D ** 2), matvec=transferRightHandle)

    # calculate fixed point
    _, r = eigs(transferRight, k=1, which='LM')

    return r.reshape(D, D)


def normaliseFixedPoints(rhoL, rhoR):
    # function that normalises given left and right fixed points
    # such that they trace to unity (interpretation as density matrix)
    # returns (rhoL, rhoR)

    trace = np.einsum('ij,ji->', rhoL, rhoR)
    # trace = np.trace(rhoL*rhoR) # might work as well/be faster than einsum?
    norm = np.sqrt(trace)
    return rhoL/norm, rhoR/norm


def qrPositive(A):
    """
    Function that implements a QR decomposition of a matrix A,
    such that A = QR
    input: A --- matrix (M, N)
    output: Q --- matrix (M, N), isometry
            R --- matrix (N, N), upper triangular, positive diagonal elements
    """

    M, N = A.shape

    # QR decomposition, scipy conventions: Q.shape = (M, M), R.shape = (M, N)
    Q, R = qr(A)

    # Throw out zeros under diagonal: Q.shape = (M, N), R.shape = (N, N)
    Q = Q[:, :N]
    R = R[:N, :]

    # extract signs and multiply with signs on diagonal
    diagSigns = np.diag(np.sign(np.diag(R)))
    Q = np.dot(Q, diagSigns)
    R = np.dot(diagSigns, R)

    return Q, R


def lqPositive(A):
    """
    Function that implements a LQ decomposition of a matrix A,
    such that A = LQ
    input: A --- matrix (M, N)
    output: L --- matrix (M, M), upper triangular, positive diagonal elements
            Q --- matrix (M, N), isometry
    """

    M, N = A.shape

    # LQ decomposition: scipy conventions: Q.shape = (N, N), L.shape = (M, N)
    L, Q = rq(A)

    # Throw out zeros under diagonal: Q.shape = (M, N), L.shape = (M, M)
    Q = Q[-M:, :]
    L = L[:, -M:]

    # Extract signs and multiply with signs on diagonal
    diagSigns = np.diag(np.sign(np.diag(L)))
    Q = np.dot(diagSigns, Q)
    L = np.dot(L, diagSigns)

    return L, Q


def leftOrthonormal(A, L0=None, tol=1e-14, maxIter=1e5):
    """
    Function that brings MPS into left-orthonormal gauge,
    such that -L-A- = -Al-L-
    input: A --- MPSTensor (D, d, D)
            L0 --- initial guess for L
            tol --- convergence tolerance
            maxIter --- max amount of steps
    output: L --- Matrix (D, D) gauges A to Al
            Al --- MPSTensor (D, d, D) left-orthonormal, -conj(Al)=Al- = --
    """

    D = A.shape[0]
    d = A.shape[1]
    i = 1

    # Random guess for L0 if none specified
    if not L0:
        L0 = np.random.rand(D, D)

    # Normalise L0
    L0 = L0 / np.linalg.norm(L0)

    # Initialise loop
    Al, L = qrPositive(np.resize(np.einsum('ik,ksj->isj', L0, A), (D * d, D)))
    L = L / np.linalg.norm(L)
    convergence = np.linalg.norm(L - L0)

    # Decompose L*A until L converges
    while convergence > tol:
        # calculate LA and decompose
        Al, Lnew = qrPositive(np.resize(np.einsum('ik,ksj->isj', L, A), (D * d, D)))

        # normalise new L
        Lnew = Lnew / np.linalg.norm(Lnew)  # only necessary when working with unnormalised MPS?

        # calculate convergence criterium
        convergence = np.linalg.norm(Lnew - L)
        L = Lnew

        # check if iterations exceeds maxIter
        if i > maxIter:
            print("Warning, decomposition has not converged ", convergence)
            break
        i += 1

    return L, np.resize(Al, (D, d, D))


def rightOrthonormal(A, R0=None, tol=1e-14, maxIter=1e5):
    """
    Function that brings MPS into right-orthonormal gauge,
    such that -A-R- = -R-Ar-
    input: A --- MPSTensor (D, d, D)
            R0 --- initial guess for R
            tol --- convergence tolerance
            maxIter --- max amount of steps
    output: R --- Matrix (D, D) gauges A to Al
            Ar --- MPSTensor (D, d, D) right-orthonormal, -conj(Ar)=Ar- = --
    """
    # function that brings MPS A into right orthonormal gauge, such that
    # A * R = R * A_R
    # returns (R, A_R)

    D = A.shape[0]
    d = A.shape[1]
    i = 1

    # Random guess for  R0 if none specified
    if not R0:
        R0 = np.random.rand(D, D)

    # Normalise R0
    R0 = R0 / np.linalg.norm(R0)

    # Initialise loop
    R, Ar = lqPositive(np.resize(np.einsum('ijk,kl->ijl', A, R0), (D, D * d)))
    R = R / np.linalg.norm(R)
    convergence = np.linalg.norm(R - R0)

    # Decompose A*R until R converges
    while convergence > tol:
        # calculate AR and decompose
        Rnew, Ar = lqPositive(np.resize(np.einsum('ijk,kl->ijl', A, R), (D, D * d)))

        # normalise new R
        Rnew = Rnew / np.linalg.norm(Rnew)  # only necessary when working with unnormalised MPS

        # calculate convergence criterium
        convergence = np.linalg.norm(Rnew - R)
        R = Rnew

        # check if iterations exceeds maxIter
        if i > maxIter:
            print("Warning, decomposition has not converged ", convergence)
            break
        i += 1

    return R, np.resize(Ar, (D, d, D))


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



def Heisenberg(Jx, Jy, Jz, h):
    """
    Function to implement spin 1 Heisenberg hamiltonian
    :param Jx: coupling in x-direction
    :param Jy: coupling in y-direction
    :param Jz: coupling in z-direction
    :param h: magnetic coupling
    :return: H: (3, 3, 3, 3) tensor (top left, top right, bottom left, bottom right)
    """
    Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    Sy = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]) * 1.0j /np.sqrt(2)
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    I = np.identity(3)

    return -Jx*np.einsum('ij,kl->ikjl', Sx, Sx)-Jy*np.einsum('ij,kl->ikjl',Sy, Sy)-Jz*np.einsum('ij,kl->ikjl', Sz, Sz) \
            - h*np.einsum('ij,kl->ikjl', I, Sz) - h*np.einsum('ij,kl->ikjl', Sz, I)



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


def rightHandle(A, v):
    # function that implements the action of a transfer matrix defined by A
    # on a right vector of dimension D**2 v (top - bottom)
    # returns a vector of dimension D**2 (top - bottom)

    D = A.shape[2]

    # contraction sequence: contract A with v, then with Abar
    newV = np.einsum('ijk,kl->ijl', A, v.reshape((D,D)))
    newV = np.einsum('ijk,ljk->il', newV, np.conj(A))

    return np.reshape(newV, D**2)


def leftHandle(A,v):
    # function that implements the action of a transfer matrix defined by A
    # on a left vector of dimension D**2 v (bottom - top)
    # returns a vector of dimension D**2 (bottom - top)

    D = A.shape[0]

    # contraction sequence: contract A with v, then with Abar
    newV = np.einsum('ijk,li->ljk', A, v.reshape((D,D)))
    newV = np.einsum('ljk,ljm->mk', newV, np.conj(A))
    return np.reshape(newV, D**2)


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
    path1 = 'einsum_path', (0, 5), (0, 1, 2, 3, 4)
    transfer_Left = LinearOperator((D**2, D**2), matvec=partial(leftHandle_, A, r, l))
    x = np.einsum('ijk,klm,jlqo,rqp,pon,ri->mn', A, A, H, np.conj(A), np.conj(A), l, optimize=path1)
    x = np.reshape(x, D**2)
    Lh = gmres(transfer_Left, x)[0]

    path2 = 'einsum_path', (1, 5), (0, 1, 2, 3, 4)
    transfer_Right = LinearOperator((D**2, D**2), matvec=partial(rightHandle_, A, r, l))
    x = np.einsum('ijk,klm,jlqo,rqp,pon,mn->ri', A, A, H, np.conj(A), np.conj(A), r, optimize=path2)
    x = np.reshape(x, D**2)
    Rh = gmres(transfer_Right, x.reshape(D**2))[0]

    Lh = np.reshape(Lh, (D, D))
    Rh = np.reshape(Rh, (D, D))
    ###########
    #FIRST TERM
    ###########
    path3 = 'einsum_path', (0, 4), (0, 3), (0, 1, 2, 3)
    first = np.einsum('ijk,klm,jlqo,pon,ri,mn->rqp', A, A, H, np.conj(A), l, r, optimize=path2)
    ###########
    #SECOND TERM
    ###########
    path3 = 'einsum_path', (0, 4), (0, 3), (0, 1, 2, 3)
    second = np.einsum('ijk,klm,jlqo,rqp,ri,mn->pon', A, A, H, np.conj(A), l, r, optimize=path3)

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


def energyDensity(A, h):
    """
    Function to calculate energy density and gradient of MPS A with, using Hamiltonian H
    :param A: MPS tensor (D, d, D)
    :param h: Hamiltonian operator (d, d, d, d)
    :return e: Energy density (real scalar)
    :return g: Gradient of energy density evaluated @A
    """

    d = A.shape[1]

    # normalise the input MPS
    A = normaliseMPS(A)

    # calculate fixed points
    l, r = leftFixedPoint(A), rightFixedPoint(A)
    l, r = normaliseFixedPoints(l, r)

    # calculate energy density
    e = twoSiteUniform(h, A, l, r)

    # check if real!
    if np.imag(e) > 1e-10:
        print("complex energy? ", e)
    e = np.real(e)

    # renormalise Hamiltonian
    hTilde = h - e * np.einsum("ik,jl->ijkl", np.identity(d), np.identity(d))

    # calculate gradient
    g = Gradient(hTilde, A, l, r)

    return e, g

def energyWrapper(H, D, d, varA):
    Areal = (varA[:D**2 *d]).reshape(D, d, D)
    Acomplex = (varA[D**2*d:]).reshape(D, d, D)
    A = Areal + 1j*Acomplex
    e, g = energyDensity(A, H)
    g = np.concatenate((np.real(g).reshape(-1), np.imag(g).reshape(-1)))
    return e, g