import numpy as np
from scipy.linalg import rq, qr, svd, polar, sqrtm
from scipy.sparse.linalg import eigs, LinearOperator, gmres
from scipy.optimize import minimize
from functools import partial
from ncon import ncon


def createMPS(bondDimension, physDimension):
    # function to create a random MPS tensor for some bondDimension and physical dimension.
    # returns a 3-legged tensor (leftLeg - physLeg - rightLeg)

    return np.random.rand(bondDimension, physDimension, bondDimension) \
        + 1j*np.random.rand(bondDimension, physDimension, bondDimension)


def createTransfer(A):
    # function to return a transfer matrix starting from a given MPS tensor.
    # returns a 4-legged tensor (topLeft - bottomLeft - topRight - bottomRight)

    return np.einsum('isk,jsl->ijkl', A, np.conj(A))


def leftFixedPoint(A):
    """
    Function to determine the left fixed point of a given MPS tensor using O(D^3) algorithm
    input: A --- (D, d, D) MPStensor
    output: l --- (D, D) leftFixedPointTensor (bottom-top)
            lam --- scalar eigenvalue of leftFixedPointTensor
    """

    D = A.shape[0]

    # construct transfer matrix handle and cast to LinearOperator
    transferLeftHandle = lambda v: np.reshape(
        ncon((A, np.conj(A), v.reshape((D, D))), ([1, 2, -2], [3, 2, -1], [3, 1]))
        , D**2)
    transferLeft = LinearOperator((D ** 2, D ** 2), matvec=transferLeftHandle)

    # calculate fixed point
    lam, l = eigs(transferLeft, k=1, which='LM')

    return lam, l.reshape(D, D)


def rightFixedPoint(A):
    """
    Function to determine the right fixed point of a given MPS tensor using O(D^3) algorithm
    input: A --- (D, d, D) MPStensor
    output: r --- (D, D) rightFixedPointTensor (top-bottom)
            lam --- scalar eigenvalue of rightFixedPointTensor
    """

    D = A.shape[0]

    # construct transfer matrix handle and cast to LinearOperator
    transferRightHandle = lambda v: np.reshape(
        ncon((A, np.conj(A), v.reshape((D,D))), ([-1, 2, 1], [-2, 2, 3], [1, 3]))
        , D ** 2)
    transferRight = LinearOperator((D ** 2, D ** 2), matvec=transferRightHandle)

    # calculate fixed point
    lam, r = eigs(transferRight, k=1, which='LM')

    return lam, r.reshape(D, D)


def normaliseFixedPoints(l, r):
    # function that normalises given left and right fixed points
    # such that they trace to unity (interpretation as density matrix)
    # returns (l, r)

    trace = np.trace(l@r)
    # trace = np.trace(rhoL*rhoR) # might work as well/be faster than einsum?
    norm = np.sqrt(trace)
    return l/norm, r/norm


def normaliseMPS(A):
    """
    Function to normalise a given MPS tensor using O(D^3) algorithm
    input: A --- (D, d, D) MPStensor
    output: A --- (D, d, D) normalised MPStensor
            l --- (D, D) leftFixedPointTensor (bottom-top)
            r --- (D, D) rightFixedPointTensor (top-bottom)
    """

    # calculate left and right fixed points of transfer matrix
    lam, l = leftFixedPoint(A)
    r = rightFixedPoint(A)[1]
    
    # normalise MPS tensor
    A /= np.sqrt(lam)
    
    # normalise fixed points
    l, r = normaliseFixedPoints(l, r)

    return A, l, r


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
    if L0 is None:
        L0 = np.random.rand(D, D)

    # Normalise L0
    L0 = L0 / np.linalg.norm(L0)

    # Initialise loop
    Al, L = qrPositive(np.reshape(ncon((L0, A), ([-1, 1], [1, -2, -3])), (D * d, D)))
    L = L / np.linalg.norm(L)
    convergence = np.linalg.norm(L - L0)

    # Decompose L*A until L converges
    while convergence > tol:
        # calculate LA and decompose
        Al, Lnew = qrPositive(np.reshape(ncon((L, A), ([-1, 1], [1, -2, -3])), (D * d, D)))

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

    return L, Al.reshape((D, d, D))


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

    D = A.shape[0]
    d = A.shape[1]
    i = 1

    # Random guess for  R0 if none specified
    if R0 is None:
        R0 = np.random.rand(D, D)

    # Normalise R0
    R0 = R0 / np.linalg.norm(R0)

    # Initialise loop
    R, Ar = lqPositive(np.reshape(ncon((A, R0), ([-1, -2, 1], [1, -3])), (D, D * d)))
    R = R / np.linalg.norm(R)
    convergence = np.linalg.norm(R - R0)

    # Decompose A*R until R converges
    while convergence > tol:
        # calculate AR and decompose
        Rnew, Ar = lqPositive(np.reshape(ncon((A, R), ([-1, -2, 1], [1, -3])), (D, D * d)))

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


def mixedCanonical(A, L0=None, R0=None, tol=1e-14, maxIter=1e5):
    """
    Function that brings MPS into mixed gauge,
    such that -Al-C- = -C-Ar- = Ac
    input:  A --- MPSTensor (D, d, D)
            L0 --- initial guess for L
            R0 --- initial guess for R
            tol --- convergence tolerance
            maxIter --- max amount of steps
    output: Al --- MPSTensor (D, d, D) left-orthonormal, -conj(Al)=Al- = --
            Ar --- MPSTensor (D, d, D) right-orthonormal, -conj(Ar)=Ar- = --
            Ac --- MPSTensor (D, d, D) center tensor
            C --- Matrix (D, D) center matrix, -Al-C- = -C-Ar- = Ac
    """

    D = A.shape[0]

    # Random guess for  L0 if none specified
    if L0 is None:
        L0 = np.random.rand(D, D)

    # Random guess for  R0 if none specified
    if R0 is None:
        R0 = np.random.rand(D, D)
    
    # Compute left and right orthonormal forms
    L, Al = leftOrthonormal(A, L0, tol, maxIter)
    R, Ar = rightOrthonormal(A, R0, tol, maxIter)
    
    # center matrix C is matrix multiplication of L and R
    C = L @ R
    
    # singular value decomposition to diagonalise C
    U, S, Vdag = svd(C)
    C = np.diag(S)

    # absorb corresponding unitaries in Al and Ar
    Al = ncon((np.conj(U).T, Al, U), ([-1, 1], [1, -2, 2], [2, -3]))
    Ar = ncon((Vdag, Ar, np.conj(Vdag).T), ([-1, 1], [1, -2, 2], [2, -3]))
    
    # normalise center matrix
    nrm = np.trace(C @ np.conj(C).T)
    C /= np.sqrt(nrm)

    # compute center MPS tensor
    Ac = ncon((Al, C), ([-1, -2, 1], [1, -3]))
        
    return Al, Ar, Ac, C


def entanglementSpectrum(Al, Ar, L, R, truncate=0):
    #Al and Ar are left and right MPS tensors
    #L and R bring A in left or right form respectively
    #find Schmidt coefficients
    #calculate bipartite entanglement entropy
    #apply truncation if desired

    #center matrix C is matrix multiplication of L and R
    C = L @ R

    #singular value decomposition
    U, S , Vdag = svd(C)

    #for well defined l and r, normalisation probably not necessary but just in case
    S = S / S[0]

    #apply a truncation step keep 'truncate' singular values
    if truncate:
        S = S[:truncate]
        U = U[:, :truncate]
        Vdag = Vdag[:truncate, :]

    # absorb corresponding unitaries in Al and Ar
    Al = ncon((np.conj(U).T, Al, U), ([-1, 1], [1, -2, 2], [2, -3]))
    Ar = ncon((Vdag, Ar, np.conj(Vdag).T), ([-1, 1], [1, -2, 2], [2, -3]))
    
    # construct and normalise center matrix
    C = np.diag(S)
    nrm = np.trace(C @ np.conj(C).T)
    C /= np.sqrt(nrm)

    # compute center MPS tensor
    Ac = ncon((Al, C), ([-1, -2, 1], [1, -3]))

    # calculate entropy using singular values
    entropy = -np.sum(S**2*np.log(S**2))

    return Al, Ar, Ac, C, entropy


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
    Sy = np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]]) * 1.0j /np.sqrt(2)
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    I = np.eye(3)

    return -Jx*ncon((Sx, Sx), ([-1, -3], [-2, -4]))-Jy*ncon((Sy, Sy), ([-1, -3], [-2, -4]))-Jz*ncon((Sz, Sz), ([-1, -3], [-2, -4])) \
            - h*ncon((I, Sz), ([-1, -3], [-2, -4])) - h*ncon((Sz, I), ([-1, -3], [-2, -4]))


def oneSiteUniform(O, A, l, r):
    # determine expectation value of one-body operator in uniform gauge
    return ncon((l, r, A, np.conj(A), O), ([4, 1], [3, 6], [1, 2, 3], [4, 5, 6], [2, 5]))


def oneSiteMixed(O, Ac):
    # determine expectation value of one-body operator in mixed gauge
    return ncon((Ac, np.conj(Ac), O), ([1, 2, 3], [1, 4, 3], [2, 4]), order=[2, 1, 3, 4])


def twoSiteUniform(O, A, l, r):
    # calculate the expectation value of the two-site operator O (top left - top right - bottom left - bottom right)
    return ncon((l, r, A, A, np.conj(A), np.conj(A), O), ([6, 1], [5, 10], [1, 2, 3], [3, 4, 5], [6, 7, 8], [8, 9, 10], [2, 4, 7, 9]))


def twoSiteMixed(O, Ac, Ar):
    # calculate the expectation value of the two-site operator O (top left - top right - bottom left - bottom right)
    return ncon((Ac, Ar, np.conj(Ac), np.conj(Ar), O), ([1, 2, 3], [3, 4, 5], [1, 6, 7], [7, 8, 5], [2, 4, 6, 8]), order=[3, 2, 4, 1, 6, 5, 8, 7])


def transferLeft(A, v):
    # function that implements the action of a transfer matrix defined by an MPS tensor A
    # on a left matrix v of dimension (D, D) (bottom - top), given as a vector of size (D**2)
    # returns a matrix of dimension (D, D) (bottom - top), given as a vector of size (D**2)

    D = A.shape[0]

    return np.reshape(ncon((v.reshape((D,D)), A, np.conj(A)), ([3, 1], [1, 2, -2], [3, 2, -1])), D**2)


def transferRight(A, v):
    # function that implements the action of a transfer matrix defined by A
    # on a right vector of dimension D**2 v (top - bottom)
    # returns a vector of dimension D**2 (top - bottom)

    D = A.shape[0]
    
    return np.reshape(ncon((A, np.conj(A), v.reshape((D,D))), ([-1, 2, 1], [-2, 2, 3], [1, 3])), D ** 2)


def transferRegularLeft(A, l, r, v):
    # function that implements the action of 1 - (T - outer(r,l))
    # on a left matrix v of dimension (D, D) (bottom - top), given as a vector of size (D**2)
    # returns a matrix of dimension (D, D) (bottom - top), given as a vector of size (D**2)

    D = A.shape[0]
    v_T = transferLeft(A, v)
    v_rl = np.trace(v.reshape((D, D)) @ r) * l
    return v - v_T + v_rl.reshape(D**2)


def transferRegularRight(A, l, r, v):
    # function that implements the action of 1-T + outer(r,l)
    # on a left matrix v of dimension (D, D) (top - bottom), given as a vector of size (D**2)
    # returns a a matrix of dimension (D, D) (top - bottom), given as a vector of size (D**2)

    D = A.shape[0]
    v_T = transferRight(A, v)
    v_rl = np.trace(l @ v.reshape((D, D))) * r
    return v - v_T + v_rl.reshape(D**2)


def energyGradient(H, A, l, r):
    """
    Function to determine the gradient of H @MPS A
    :param H: (d, d, d, d) (regularised) hamiltonian density operator
    :param A: (D, d, D) MPS tensor
    :param l: (D, D) left fixed point (normalised!)
    :param r: (D, D) right fixed point (normalised!)
    :return:
    """

    # a rank 3 tensor, equation (116) in the notes
    # consists of 4 terms
    # have to solve x = y(1-T_) where T_ = T - l \otimes r is the regularised transfer matrix
    # don't naively construct (1-T_) because all these objects have 4D legs. As before describe how this operator works on a vector y
    # create function handle instead of D**2 matrix
    D = A.shape[0]

    # first center term
    centerTerm1 = ncon((l, r, A, A, np.conj(A), H), ([6, 1], [5, -3], [1, 3, 2], [2, 4, 5], [6, 7, -1], [3, 4, 7, -2]))
    
    # second center term
    centerTerm2 = ncon((l, r, A, A, np.conj(A), H), ([-1, 1], [5, 7], [1, 3, 2], [2, 4, 5], [-3, 6, 7], [3, 4, -2, 6]))

    # left environment term
    xL = ncon((l, A, A, np.conj(A), np.conj(A), H), ([5, 1], [1, 3, 2], [2, 4, -2], [5, 6, 7], [7, 8, -1], [3, 4, 6, 8]))
    transfer_Left = LinearOperator((D**2, D**2), matvec=partial(transferRegularLeft, A, l, r))
    Lh = gmres(transfer_Left, xL.reshape(D ** 2))[0].reshape((D,D))
    leftEnvTerm = ncon((Lh, A, r), ([-1, 1], [1, -2, 2], [2, -3]))

    # right environment term
    xR =  ncon((r, A, A, np.conj(A), np.conj(A), H), ([4, 5], [-1, 2, 1], [1, 3, 4], [-2, 8, 7], [7, 6, 5], [2, 3, 8, 6]))
    transfer_Right = LinearOperator((D**2, D**2), matvec=partial(transferRegularRight, A, l, r))
    Rh = gmres(transfer_Right, xR.reshape(D**2))[0].reshape((D,D))
    rightEnvTerm = ncon((Rh, A, l), ([1, -3], [2, -2, 1], [-1, 2]))

    return 2 * (centerTerm1 + centerTerm2 + leftEnvTerm + rightEnvTerm)


def energyDensity(A, H):
    """
    Function to calculate energy density and gradient of MPS A with, using Hamiltonian H
    :param A: MPS tensor (D, d, D)
    :param H: Hamiltonian operator (d, d, d, d)
    :return e: Energy density (real scalar)
    :return g: Gradient of energy density evaluated @A
    """

    d = A.shape[1]

    # normalise the input MPS
    A, l, r = normaliseMPS(A)

    # calculate energy density
    e = np.real(twoSiteUniform(H, A, l, r))

    # regularise Hamiltonian
    Htilde = H - e * ncon((np.eye(d), np.eye(d)), ([-1, -3], [-2, -4]))

    # calculate gradient of energy
    g = energyGradient(Htilde, A, l, r)

    return e, g


def energyWrapper(H, D, d, varA):
    """
    Wrapper around energyDensity function that takes complex MPS tensor of
    size (D, d, D) as a real vector of size (2 * D ** 2 * d) and returns the
    complex gradient tensor of size (D, d, D) as a real vector of size
    (2 * D ** 2 * d)
    
    Parameters
    ----------
    H : np.array(d, d, d, d)
        Two-site Hamiltonian operator.
    D : int
        Bond dimension of MPS.
    d : int
        Physical dimension of sites in spin chain.
    varA : np.array(2 * D ** 2 * d)
        Real vector of size (2 * D ** 2 * d) that represents a complex MPS
        tensor of size (D, d, D).

    Returns
    -------
    e : float
        Energy density (real scalar).
    g : np.array(2 * D ** 2 * d)
        Real vector of size (2 * D ** 2 * d) that represents the complex
        energy gradient tensor of size (D, d, D).
    """
    Areal = (varA[:D**2 *d]).reshape(D, d, D)
    Acomplex = (varA[D**2*d:]).reshape(D, d, D)
    A = Areal + 1j*Acomplex
    e, g = energyDensity(A, H)
    g = np.concatenate((np.real(g).reshape(-1), np.imag(g).reshape(-1)))
    return e, g


#### functions for vumps

def rightEnvMixed(Ar, C, Htilde, delta):
    '''
    :param Ar:
    :param L:
    :param R:
    :param hTilde:
    :return:
    '''
    D = Ar.shape[0]
    
    xR =  ncon((Ar, Ar, np.conj(Ar), np.conj(Ar), Htilde), ([-1, 2, 1], [1, 3, 4], [-2, 7, 6], [6, 5, 4], [2, 3, 7, 5]))
    # !!!!!!!!!!!!!!!!!!!!MADE A MISTAKE HERE BEFORE BUT PROCEDURE STILL CONVERGED!!!!!!!!!!!!!!!!
    # transfer_Right = LinearOperator((D**2, D**2), matvec=partial(transferRegularRight, Ar, C @ np.conj(C).T, np.eye(D)))
    transfer_Right = LinearOperator((D**2, D**2), matvec=partial(transferRegularRight, Ar, np.conj(C).T @ C, np.eye(D)))
    Rh = gmres(transfer_Right, xR.reshape(-1), tol=delta/10)[0]
    
    return Rh.reshape(D, D)


#left environment vumps
def leftEnvMixed(Al, C, Htilde, delta):
    '''
    :param Al:
    :param L:
    :param R:
    :param hTilde:
    :return:
    '''
    D = Al.shape[0]
    xL =  ncon((Al, Al, np.conj(Al), np.conj(Al), Htilde), ([4, 2, 1], [1, 3, -2], [4, 5, 6], [6, 7, -1], [2, 3, 5, 7]))
    transfer_Left = LinearOperator((D**2, D**2), matvec=partial(transferRegularLeft, Al, np.eye(D), C @ np.conj(C).T))
    Lh = gmres(transfer_Left, xL.reshape(-1), tol=delta/10)[0]

    return Lh.reshape(D, D)


def H_Ac(v, Al, Ar, Rh, Lh, Htilde):
    '''
    :param v:
    :param Al:
    :param Ar:
    :param Rh:
    :param Lh:
    :param hTilde:
    :return:
    '''
    centerTerm1 = ncon((Al, v, np.conj(Al), Htilde), ([4, 2, 1], [1, 3, -3], [4, 5, -1], [2, 3, 5, -2]))
    centerTerm2 = ncon((v, Ar, np.conj(Ar), Htilde), ([-1, 2, 1], [1, 3, 4], [-3, 5, 4], [2, 3, -2, 5]))
    leftEnvTerm = ncon((Lh, v), ([-1, 1], [1, -2, -3]))
    rightEnvTerm = ncon((v, Rh), ([-1, -2, 1], [1, -3]))
    
    return centerTerm1 + centerTerm2 + leftEnvTerm + rightEnvTerm

def H_C(v, Al, Ar, Rh, Lh, Htilde):
    '''
    :param v:
    :param Al:
    :param Ar:
    :param Rh:
    :param Lh:
    :param hTilde:
    :return:
    '''
    centerTerm = ncon((Al, v, Ar, np.conj(Al), np.conj(Ar), Htilde), ([5, 3, 1], [1, 2], [2, 4, 7], [5, 6, -1], [-2, 8, 7], [3, 4, 6, 8]))
    leftEnvTerm = Lh @ v
    rightEnvTerm = v @ Rh

    return centerTerm + leftEnvTerm + rightEnvTerm


def calcNewCenter(Al, Ar, Ac, C, Lh, Rh, Htilde, delta):
    '''
    :param Al:
    :param Ar:
    :param L:
    :param R:
    :param Lh:
    :param Rh:
    :param hTilde:
    :return:
    '''
    D = Al.shape[0]
    d = Al.shape[1]
    handleAc = lambda v: (H_Ac(v.reshape(D, d, D), Al, Ar, Rh, Lh, Htilde)).reshape(-1)
    handleAc = LinearOperator((D ** 2 * d, D ** 2 * d), matvec=handleAc)
    handleC = lambda v: (H_C(v.reshape(D, D), Al, Ar, Rh, Lh, Htilde)).reshape(-1)
    handleC = LinearOperator((D ** 2, D ** 2), matvec=handleC)
    _, AcPrime = eigs(handleAc, k=1, which="SR", v0=Ac.reshape(-1), tol=delta/10)
    _, cPrime = eigs(handleC, k=1, which="SR", v0=C.reshape(-1), tol=delta/10)
    return AcPrime.reshape((D,d,D)), cPrime.reshape((D,D))


def minAcC(AcPrime, cPrime):
    '''
    :param AcPrime:
    :param cPrime:
    :return:
    '''
    D = AcPrime.shape[0]
    d = AcPrime.shape[1]
    UlAc, _ = polar(AcPrime.reshape(D*d,D))
    UlC, _ = polar(cPrime)
    Al = (UlAc @ np.conj(UlC).T).reshape(D, d, D)
    _, Ar = rightOrthonormal(Al)
    Ac = AcPrime
    C = cPrime
    nrm = np.trace(C @ np.conj(C).T)
    Ac = Ac / np.sqrt(nrm)
    C = C / np.sqrt(nrm)
    return Al, Ar, Ac, C


def delta(d, n):
    out = np.zeros( (d,) * n )
    out[ tuple([np.arange(d)] * n) ] = 1
    return out


def O(beta, J):
    c, s = np.sqrt(np.cosh(beta*J)), np.sqrt(np.sinh(beta*J))
    #test
    #Q = np.array([[np.exp(beta), np.exp(-beta)],[np.exp(-beta), np.exp(beta)]])
    #Q_sqrt_ = sqrtm(Q)
    Q_sqrt = 1/2 * np.array([[c+s, c-s],[c-s, c+s]])
    O = ncon((Q_sqrt, Q_sqrt, Q_sqrt, Q_sqrt, delta(2,4)), ([-1,1], [-2,2], [-3,3], [-4,4], [1,2,3,4]))
    return O


def M(beta, J):
    S_z = np.array([[1,0],[0,-1]])
    c, s = np.sqrt(np.cosh(beta*J)), np.sqrt(np.sinh(beta*J))
    Q_sqrt = 1/np.sqrt(2) * np.array([[c+s, c-s],[c-s, c+s]])
    delta_new = ncon((S_z, delta(2,4)), ([-1,1], [1,-2,-3,-4]))
    M = ncon((Q_sqrt, Q_sqrt, Q_sqrt, Q_sqrt, delta_new), ([-1,1], [-2,2], [-3,3], [-4,4], [1,2,3,4]))
    return M


def free_energy_density(beta, J):
    Lambda=1
    return -np.log(Lambda)