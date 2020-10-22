import numpy as np
from scipy.linalg import rq, qr, svd, polar
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


def leftFixedPoint(A):
    """
    Function to determine the left fixed point of a given MPS tensor using O(D^3) algorithm
    input: A --- (D, d, D) MPStensor
    output: l --- (D, D) leftFixedPointTensor (bottom-top)
            lam --- scalar eigenvalue of leftFixedPointTensor
    """

    D = A.shape[0]

    # set optimal contraction sequence
    path = ['einsum_path', (0, 2), (0, 1)]

    # calculate transfer matrix handle and cast to LinearOperator
    transferLeftHandle = lambda v: np.reshape(
        np.einsum('ijk,ljm,li->mk', A, np.conj(A), v.reshape((D, D)), optimize=path), D ** 2)
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

    # set optimal contraction sequence
    path = ['einsum_path', (0, 2), (0, 1)]

    # calculate transfer matrix handle and cast to LinearOperator
    transferRightHandle = lambda v: np.reshape(
        np.einsum('ijk,ljm,km->il', A, np.conj(A), v.reshape((D, D)), optimize=path), D ** 2)
    transferRight = LinearOperator((D ** 2, D ** 2), matvec=transferRightHandle)

    # calculate fixed point
    lam, r = eigs(transferRight, k=1, which='LM')

    return lam, r.reshape(D, D)


def normaliseFixedPoints(l, r):
    # function that normalises given left and right fixed points
    # such that they trace to unity (interpretation as density matrix)
    # returns (l, r)

    trace = np.einsum('ij,ji->', l, r)
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

    D = A.shape[0]
    d = A.shape[1]
    i = 1

    # Random guess for  R0 if none specified
    if R0 is None:
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
    R, Ar = rightOrthonormal(A, L0, tol, maxIter)
    
    # center matrix C is matrix multiplication of L and R
    C = L @ R
    
    # singular value decomposition to diagonalise C
    U, S, Vdag = svd(C)
    C = np.diag(S)

    # absorb corresponding unitaries in Al and Ar
    Al = np.einsum('ij,jkl,lm->ikm', np.conj(U).T, Al, U)
    Ar = np.einsum('ij,jkl,lm->ikm', Vdag, Ar, np.conj(Vdag).T)
    
    # normalise center matrix
    nrm = np.trace(C @ np.conj(C).T)
    C /= np.sqrt(nrm);

    # compute center MPS tensor
    Ac = np.einsum('ijk,kl->ijl', Al, C)
        
    return Al, Ar, Ac, C


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
    I = np.eye(3)

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
    path = 'einsum_path', (0, 5), (0, 4), (0, 1, 2, 3, 4)
    return np.einsum('ijk,klm,jlqo,rqp,pon,ri,mn', A, A, H, np.conj(A), np.conj(A), l, r, optimize=path)


def twoSiteMixed(H, Ac, Ar):
    #calculate the expectation value of the hamiltonian H (top left - top right - bottom left - bottom right)
    #in mixed canonical form that acts on two sites, contraction done from right to left
    #case where Ac on left legs of H
    # kjlipmno
    return np.einsum('ijk,klm,jlpn,ipo,onm', Ac, Ar, H, np.conj(Ac), np.conj(Ar), optimize=True)


def transferRight(A, v):
    # function that implements the action of a transfer matrix defined by A
    # on a right vector of dimension D**2 v (top - bottom)
    # returns a vector of dimension D**2 (top - bottom)

    D = A.shape[0]

    # contraction sequence: contract A with v, then with Abar
    newV = np.einsum('ijk,kl->ijl', A, v.reshape((D,D)))
    newV = np.einsum('ijk,ljk->il', newV, np.conj(A))

    return np.reshape(newV, D**2)


def transferLeft(A, v):
    # function that implements the action of a transfer matrix defined by A
    # on a left vector of dimension D**2 v (bottom - top)
    # returns a vector of dimension D**2 (bottom - top)

    D = A.shape[0]

    # contraction sequence: contract A with v, then with Abar
    newV = np.einsum('ijk,li->ljk', A, v.reshape((D, D)))
    newV = np.einsum('ljk,ljm->mk', newV, np.conj(A))
    return np.reshape(newV, D**2)


def transferRegularLeft(A, l, r, v):
    # function that implements the action of 1-T + outer(r,l)
    # on a left vector of dimension D**2 v (bottom - top)
    # returns a vector of dimension D**2 (bottom - top)

    D = A.shape[0]
    v_T = transferLeft(A, v)
    v_rl = np.trace(v.reshape((D, D))@r) * l
    return v - v_T + np.reshape(v_rl, D**2)


def transferRegularRight(A, l, r, v):
    # function that implements the action of 1-T + outer(r,l)
    # on a left vector of dimension D**2 v (bottom - top)
    # returns a vector of dimension D**2 (bottom - top)

    D = A.shape[0]
    v_T = transferRight(A,v)
    v_rl = np.trace(l@v.reshape((D,D))) * r
    return v - v_T + np.reshape(v_rl, D**2)


def energyGradient(H, A, l, r):
    """
    Function to determine the gradient of H @MPS A
    :param H: (d, d, d, d) hamiltonian density operator
    :param A: (D, d, D) MPS tensor
    :param l: (D, D) left fixed point (normalised!)
    :param r: (D, D) right fixed point (normalised!)
    :return:
    """

    # a rank 3 tensor, equation (116) in the notes
    # consists of 4 terms
    # have to solve x = y(1-T_) where T_ = createTransfer(A) - np.outer(leftFixedPoint(A), rightFixedPoint(A))
    # don't naively construct (1-T_) because all these objects have 4D legs. As before describe how this operator works on a vector y
    # create function handle instead of D**2 matrix
    D = A.shape[0]


    path1 = 'einsum_path', (0, 5), (0, 1, 2, 3, 4)

    transfer_Left = LinearOperator((D**2, D**2), matvec=partial(transferRegularLeft, A, l, r))
    x = np.einsum('ijk,klm,jlqo,rqp,pon,ri->nm', A, A, H, np.conj(A), np.conj(A), l, optimize=path1)
    x = np.reshape(x, D**2)
    Lh = gmres(transfer_Left, x)[0]

    path2 = 'einsum_path', (1, 5), (0, 1, 2, 3, 4)
    transfer_Right = LinearOperator((D**2, D**2), matvec=partial(transferRegularRight, A, l, r))
    x = np.einsum('ijk,klm,jlqo,rqp,pon,mn->ir', A, A, H, np.conj(A), np.conj(A), r, optimize=path2)
    x = np.reshape(x, D**2)
    Rh = gmres(transfer_Right, x.reshape(D**2))[0]

    Lh = np.reshape(Lh, (D, D))
    Rh = np.reshape(Rh, (D, D))
    ###########
    #FIRST TERM
    ###########
    path3 = 'einsum_path', (0, 4), (0, 3), (0, 1, 2, 3)
    first = np.einsum('ijk,klm,jlqo,pon,ri,mn->rqp', A, A, H, np.conj(A), l, r, optimize=path3)
    
    ###########
    #SECOND TERM
    ###########
    path4 = 'einsum_path', (0, 4), (0, 3), (0, 1, 2, 3)
    second = np.einsum('ijk,klm,jlqo,rqp,ri,mn->pon', A, A, H, np.conj(A), l, r, optimize=path4)

    ###########
    #THIRD TERM
    ###########
    third = np.einsum('mi,ijk,kl->mjl', l, A, Rh)

    ###########
    #FOURTH TERM
    ###########
    fourth = np.einsum('mi,ijk,kl->mjl', Lh, A, r)

    # define Lh and Rh
    return 2 * (first+second+third+fourth)


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
    A, l, r = normaliseMPS(A)

    # calculate energy density
    e = twoSiteUniform(h, A, l, r)

    # check if real!
    if np.imag(e) > 1e-14:
        print("complex energy? ", e)
    e = np.real(e)

    # renormalise Hamiltonian
    hTilde = h - e * np.einsum("ik,jl->ijkl", np.eye(d), np.eye(d))

    # calculate gradient
    g = energyGradient(hTilde, A, l, r)

    return e, g

def energyWrapper(H, D, d, varA):
    Areal = (varA[:D**2 *d]).reshape(D, d, D)
    Acomplex = (varA[D**2*d:]).reshape(D, d, D)
    A = Areal + 1j*Acomplex
    e, g = energyDensity(A, H)
    g = np.concatenate((np.real(g).reshape(-1), np.imag(g).reshape(-1)))
    return e, g

#### functions for vumps

# right environment vumps
def rightEnvMixed(Ar, C, hTilde, delta):
    '''
    :param Ar:
    :param L:
    :param R:
    :param hTilde:
    :return:
    '''
    D = Ar.shape[0]
    transfer_Right = LinearOperator((D**2, D**2), matvec=partial(transferRegularRight, Ar, C @ np.conj(C).T, np.eye(D)))
    xR = np.einsum("ijk,klm,nop,pqm,jloq->in", Ar, Ar, np.conj(Ar), np.conj(Ar), hTilde, optimize=True)
    Rh = gmres(transfer_Right, xR.reshape(-1), tol=delta/10)[0]
    
    return Rh.reshape(D, D)

#left environment vumps
def leftEnvMixed(Al, C, hTilde, delta):
    '''
    :param Al:
    :param L:
    :param R:
    :param hTilde:
    :return:
    '''
    D = Al.shape[0]
    transfer_Left = LinearOperator((D**2, D**2), matvec=partial(transferRegularLeft, Al, np.eye(D), C @ np.conj(C).T))
    xL = np.einsum("ijk,klm,ino,opq,jlnp->qm", Al, Al, np.conj(Al), np.conj(Al), hTilde, optimize=True)
    Lh = gmres(transfer_Left, xL.reshape(-1), tol=delta/10)[0]

    return Lh.reshape(D, D)

def H_Ac(v, Al, Ar, Rh, Lh, hTilde):
    '''
    :param v:
    :param Al:
    :param Ar:
    :param Rh:
    :param Lh:
    :param hTilde:
    :return:
    '''
    centerTerm1 = np.einsum("ijk,klm,ino,jlnp->opm", Al, v, np.conj(Al), hTilde, optimize=True)
    centerTerm2 = np.einsum("ijk,klm,nom,jlpo->ipn", v, Ar, np.conj(Ar), hTilde, optimize=True)
    leftEnvTerm = np.einsum("ij,jkl -> ikl", Lh, v)
    rightEnvTerm = np.einsum("ijk,kl->ijl", v, Rh)

    return centerTerm1 + centerTerm2 + leftEnvTerm + rightEnvTerm

def H_C(v, Al, Ar, Rh, Lh, hTilde):
    '''
    :param v:
    :param Al:
    :param Ar:
    :param Rh:
    :param Lh:
    :param hTilde:
    :return:
    '''
    centerTerm = np.einsum("ijk,kl,lmn,iop,qrn,jmor->pq", Al, v, Ar, np.conj(Al), np.conj(Ar), hTilde, optimize=True)
    leftEnvTerm = Lh @ v
    rightEnvTerm = v @ Rh

    return centerTerm + leftEnvTerm + rightEnvTerm

def calcNewCenter(Al, Ar, Ac, C, Lh, Rh, hTilde, delta):
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
    handleAc = lambda v: (H_Ac(v.reshape(D, d, D), Al, Ar, Rh, Lh, hTilde)).reshape(-1)
    handleAc = LinearOperator((D ** 2 * d, D ** 2 * d), matvec=handleAc)
    handleC = lambda v: (H_C(v.reshape(D, D), Al, Ar, Rh, Lh, hTilde)).reshape(-1)
    handleC = LinearOperator((D ** 2, D ** 2), matvec=handleC)
    _, AcPrime = eigs(handleAc, k=1, which="SR", v0=Ac.reshape(-1), tol=delta/10)
    AcPrime = AcPrime.reshape(D, d, D)
    _, cPrime = eigs(handleC, k=1, which="SR", v0=C.reshape(-1), tol=delta/10)
    cPrime = cPrime.reshape(D, D)
    return AcPrime, cPrime

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
    norm = np.einsum("ijk,ijk", Ac, np.conj(Ac))
    Ac /= np.sqrt(norm)
    return Al, Ar, Ac, C
