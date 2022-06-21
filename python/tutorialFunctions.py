"""
Summary of all functions used and defined in python notebooks
"""

# all necessary imports
import numpy as np
from scipy.linalg import rq, qr, svd, polar
from scipy.sparse.linalg import eigs, LinearOperator, gmres
from scipy.optimize import minimize
from functools import partial
from ncon import ncon


"""
Chapter 1
"""


def createMPS(D, d):
    """
    Returns a random complex MPS tensor.

        Parameters
        ----------
        D : int
            Bond dimension for MPS.
        d : int
            Physical dimension for MPS.

        Returns
        -------
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            normalized.
    """

    A = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)

    return normalizeMPS(A)


def createTransfermatrix(A):
    """
    Form the transfermatrix of an MPS.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        E : np.array(D, D, D, D)
            Transfermatrix with 4 legs,
            ordered topLeft-bottomLeft-topRight-bottomRight.
    """

    E = ncon((A, np.conj(A)), ([-1, 1, -3], [-2, 1, -4]))

    return E


def normalizeMPS(A):
    """
    Normalize an MPS tensor.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        Anew : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    D = A.shape[0]

    # calculate transfer matrix handle and cast to LinearOperator
    handleERight = lambda v: np.reshape(ncon((A, np.conj(A), v.reshape((D,D))), ([-1, 2, 1], [-2, 2, 3], [1, 3])),
                                        D ** 2)
    E = LinearOperator((D ** 2, D ** 2), matvec=handleERight)

    # calculate eigenvalue
    lam = eigs(E, k=1, which='LM', return_eigenvectors=False)

    Anew = A / np.sqrt(lam)

    return Anew


def leftFixedPoint(A):
    """
    Find left fixed point.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        l : np.array(D, D)
            left fixed point with 2 legs,
            ordered bottom-top.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    D = A.shape[0]

    # calculate transfer matrix handle and cast to LinearOperator
    handleELeft = lambda v: np.reshape(ncon((A, np.conj(A), v.reshape((D, D))), ([1, 2, -2], [3, 2, -1], [3, 1])), D ** 2)
    E = LinearOperator((D ** 2, D ** 2), matvec=handleELeft)

    # calculate fixed point
    _, l = eigs(E, k=1, which='LM')
    
    # reshape to matrix
    l = l.reshape((D, D))
    
    # make left fixed point hermitian explicitly
    l /= (np.trace(l) / np.abs(np.trace(l)))# remove possible phase
    l = (l + np.conj(l).T) / 2 # force hermitian
    l *= np.sign(np.trace(l)) # force positive semidefinite

    return l


def rightFixedPoint(A):
    """
    Find right fixed point.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        r : np.array(D, D)
            right fixed point with 2 legs,
            ordered top-bottom.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    D = A.shape[0]

    # calculate transfer matrix handle and cast to LinearOperator
    handleERight = lambda v: np.reshape(ncon((A, np.conj(A), v.reshape((D,D))), ([-1, 2, 1], [-2, 2, 3], [1, 3])), D ** 2)
    E = LinearOperator((D ** 2, D ** 2), matvec=handleERight)

    # calculate fixed point
    _, r = eigs(E, k=1, which='LM')
    
    # reshape to matrix
    r = r.reshape((D, D))
    
    # make right fixed point hermitian explicitly
    r /= (np.trace(r) / np.abs(np.trace(r)))# remove possible phase
    r = (r + np.conj(r).T) / 2 # force hermitian
    r *= np.sign(np.trace(r)) # force positive semidefinite
    
    return r


def fixedPoints(A):
    """
    Find normalized fixed points.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        l : np.array(D, D)
            left fixed point with 2 legs,
            ordered bottom-top.
        r : np.array(D, D)
            right fixed point with 2 legs,
            ordered top-bottom.

        Complexity
        ----------
        O(D ** 3) algorithm,
            D ** 3 contraction for transfer matrix handle.
    """

    # find fixed points
    l, r = leftFixedPoint(A), rightFixedPoint(A)

    # calculate trace
    trace = np.trace(l@r)

    return l / trace, r


def rqPos(A):
    """
    Do a RQ decomposition with positive diagonal elements for R.

        Parameters
        ----------
        A : np.array(M, N)
            Matrix to decompose.

        Returns
        -------
        R : np.array(M, M)
            Upper triangular matrix,
            positive diagonal elements.
        Q : np.array(M, N)
            Orthogonal matrix.

        Complexity
        ----------
        ~O(max(M, N) ** 3) algorithm.
    """

    M, N = A.shape

    # LQ decomposition: scipy conventions: Q.shape = (N, N), L.shape = (M, N)
    R, Q = rq(A)

    # Throw out zeros under diagonal: Q.shape = (M, N), L.shape = (M, M)
    Q = Q[-M:, :]
    R = R[:, -M:]

    # Extract signs and multiply with signs on diagonal
    diagSigns = np.diag(np.sign(np.diag(R)))
    Q = np.dot(diagSigns, Q)
    R = np.dot(R, diagSigns)

    return R, Q


def rightOrthonormalize(A, R0=None, tol=1e-14, maxIter=1e5):
    """
    Transform A to right-orthonormal gauge.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        R0 : np.array(D, D), optional
            Right gauge matrix,
            initial guess.
        tol : float, optional
            convergence criterium,
            norm(R - Rnew) < tol.
        maxIter : int
            maximum amount of iterations.

        Returns
        -------
        R : np.array(D, D)
            right gauge with 2 legs,
            ordered left-right.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal
    """

    D = A.shape[0]
    d = A.shape[1]
    tol = max(tol, 1e-14)
    i = 1

    # Random guess for R0 if none specified
    if R0 is None:
        R0 = np.random.rand(D, D)

    # Normalize R0
    R0 = R0 / np.linalg.norm(R0)

    # Initialize loop
    R, Ar = rqPos(np.reshape(ncon((A, R0), ([-1, -2, 1], [1, -3])), (D, D * d)))
    R = R / np.linalg.norm(R)
    convergence = np.linalg.norm(R - R0)

    # Decompose A*R until R converges
    while convergence > tol:
        # calculate AR and decompose
        Rnew, Ar = rqPos(np.reshape(ncon((A, R), ([-1, -2, 1], [1, -3])), (D, D * d)))

        # normalize new R
        Rnew = Rnew / np.linalg.norm(Rnew)

        # calculate convergence criterium
        convergence = np.linalg.norm(Rnew - R)
        R = Rnew

        # check if iterations exceeds maxIter
        if i > maxIter:
            print("Warning, right decomposition has not converged ", convergence)
            break
        i += 1

    return R, Ar.reshape((D, d, D))


def qrPos(A):
    """
    Do a QR decomposition with positive diagonal elements for R.

        Parameters
        ----------
        A : np.array(M, N)
            Matrix to decompose.

        Returns
        -------
        Q : np.array(M, N)
            Orthogonal matrix.
        R : np.array(N, N)
            Upper triangular matrix,
            positive diagonal elements.

        Complexity
        ----------
        ~O(max(M, N) ** 3) algorithm.
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


def leftOrthonormalize(A, L0=None, tol=1e-14, maxIter=1e5):
    """
    Transform A to left-orthonormal gauge.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        L0 : np.array(D, D), optional
            Left gauge matrix,
            initial guess.
        tol : float, optional
            convergence criterium,
            norm(R - Rnew) < tol.
        maxIter : int
            maximum amount of iterations.

        Returns
        -------
        L : np.array(D, D)
            left gauge with 2 legs,
            ordered left-right.
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal
    """

    D = A.shape[0]
    d = A.shape[1]
    tol = max(tol, 1e-14)
    i = 1

    # Random guess for L0 if none specified
    if L0 is None:
        L0 = np.random.rand(D, D)

    # Normalize L0
    L0 = L0 / np.linalg.norm(L0)

    # Initialize loop
    Al, L = qrPos(np.reshape(ncon((L0, A), ([-1, 1], [1, -2, -3])), (D * d, D)))
    L = L / np.linalg.norm(L)
    convergence = np.linalg.norm(L - L0)

    # Decompose L*A until L converges
    while convergence > tol:
        # calculate LA and decompose
        Al, Lnew = qrPos(np.reshape(ncon((L, A), ([-1, 1], [1, -2, -3])), (D * d, D)))

        # normalize new L
        Lnew = Lnew / np.linalg.norm(Lnew)

        # calculate convergence criterium
        convergence = np.linalg.norm(Lnew - L)
        L = Lnew

        # check if iterations exceeds maxIter
        if i > maxIter:
            print("Warning, left decomposition has not converged ", convergence)
            break
        i += 1

    return L, Al.reshape((D, d, D))


def mixedCanonical(A, L0=None, R0=None, tol=1e-14, maxIter=1e5):
    """
    Bring MPS tensor into mixed gauge, such that -Al-C- = -C-Ar- = Ac.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            diagonal.

        Complexity
        ----------
        O(D ** 3) algorithm.
    """

    D = A.shape[0]
    tol = max(tol, 1e-14)

    # Random guess for  L0 if none specified
    if L0 is None:
        L0 = np.random.rand(D, D)

    # Random guess for  R0 if none specified
    if R0 is None:
        R0 = np.random.rand(D, D)

    # Compute left and right orthonormal forms
    L, Al = leftOrthonormalize(A, L0, tol, maxIter)
    R, Ar = rightOrthonormalize(A, R0, tol, maxIter)

    # center matrix C is matrix multiplication of L and R
    C = L @ R

    # singular value decomposition to diagonalize C
    U, S, Vdag = svd(C)
    C = np.diag(S)

    # absorb corresponding unitaries in Al and Ar
    Al = ncon((np.conj(U).T, Al, U), ([-1, 1], [1, -2, 2], [2, -3]))
    Ar = ncon((Vdag, Ar, np.conj(Vdag).T), ([-1, 1], [1, -2, 2], [2, -3]))

    # normalize center matrix
    norm = np.trace(C @ np.conj(C).T)
    C /= np.sqrt(norm)

    # compute center MPS tensor
    Ac = ncon((Al, C), ([-1, -2, 1], [1, -3]))

    return Al, Ac, Ar, C


def entanglementSpectrum(A):
    """
    Calculate the entanglement spectrum of an MPS.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.

        Returns
        -------
        S : np.array (D,)
            Singular values of center matrix,
            representing the entanglement spectrum
        entropy : float
        entropy : float
            Entanglement entropy across a leg.
    """

    # go to mixed gauge
    _, _, _, C = mixedCanonical(A)

    # calculate entropy
    S = np.diag(C)
    entropy = -np.sum(S ** 2 * np.log(S ** 2))

    return S, entropy


def truncateMPS(A, Dtrunc):
    """
    Truncate an MPS to a lower bond dimension.

        Parameters
        ----------
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        Dtrunc : int
            lower bond dimension

        Returns
        -------
        AlTilde : np.array(Dtrunc, d, Dtrunc)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        AcTilde : np.array(Dtrunc, d, Dtrunc)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        ArTilde : np.array(Dtrunc, d, Dtrunc)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        CTilde : np.array(Dtrunc, Dtrunc)
            Center gauge with 2 legs,
            ordered left-right,
            diagonal.
    """

    Al, Ac, Ar, C = mixedCanonical(A)

    # perform SVD and truncate:
    U, S, Vdag = svd(C)
    U = U[:, :Dtrunc]
    Vdag = Vdag[:Dtrunc, :]
    S = S[:Dtrunc]

    # reabsorb unitaries
    AlTilde = ncon((np.conj(U).T, Al, U), ([-1, 1], [1, -2, 2], [2, -3]))
    ArTilde = ncon((Vdag, Ar, np.conj(Vdag).T), ([-1, 1], [1, -2, 2], [2, -3]))
    CTilde = np.diag(S)

    # renormalize
    norm = np.trace(CTilde @ np.conj(CTilde).T)
    CTilde /= np.sqrt(norm)

    AcTilde = ncon((AlTilde, CTilde), ([-1, -2, 1], [1, -3]))

    return AlTilde, AcTilde, ArTilde, CTilde


def expVal1Uniform(O, A, l=None, r=None):
    """
    Calculate the expectation value of a 1-site operator in uniform gauge.

        Parameters
        ----------
        O : np.array(d, d)
            single-site operator.
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.

        Returns
        -------
        o : complex float
            expectation value of O.
    """

    # calculate fixed points if not given
    if l is None or r is None:
        l, r = fixedPoints(A)

    # contract expectation value network
    o = ncon((l, r, A, np.conj(A), O), ([4, 1], [3, 6], [1, 2, 3], [4, 5, 6], [2, 5]))

    return o


def expVal1Mixed(O, Ac):
    """
    Calculate the expectation value of a 1-site operator in mixed gauge.

        Parameters
        ----------
        O : np.array(d, d)
            single-site operator.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauged.

        Returns
        -------
        o : complex float
            expectation value of O.
    """

    # contract expectation value network
    o = ncon((Ac, np.conj(Ac), O), ([1, 2, 3], [1, 4, 3], [2, 4]), order=[2, 1, 3, 4])

    return o


def expVal2Uniform(O, A, l=None, r=None):
    """
    Calculate the expectation value of a 2-site operator in uniform gauge.

        Parameters
        ----------
        O : np.array(d, d, d, d)
            two-site operator,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        A : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.

        Returns
        -------
        o : complex float
            expectation value of O.
    """

    # calculate fixed points if not given
    if l is None or r is None:
        l, r = fixedPoints(A)

    # contract expectation value network
    o = ncon((l, r, A, A, np.conj(A), np.conj(A), O), ([6, 1], [5, 10], [1, 2, 3], [3, 4, 5], [6, 7, 8], [8, 9, 10], [2, 4, 7, 9]))

    return o


def expVal2Mixed(O, Ac, Ar):
    """
    Calculate the expectation value of a 2-site operator in mixed gauge.

        Parameters
        ----------
        O : np.array(d, d, d, d)
            two-site operator,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauged.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right gauged.

        Returns
        -------
        o : complex float
            expectation value of O.
    """

    # contract expectation value network
    o = ncon((Ac, Ar, np.conj(Ac), np.conj(Ar), O), ([1, 2, 3], [3, 4, 5], [1, 6, 7], [7, 8, 5], [2, 4, 6, 8]), order=[3, 2, 4, 1, 6, 5, 8, 7])

    return o


"""
Chapter 2
"""


def gradCenterTerms(hTilde, A, l=None, r=None):
    """
    Calculate the value of the center terms.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        A : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        
        Returns
        -------
        term1 : np.array(D, d, D)
            first term of gradient,
            ordered left-mid-right.
        term2 : np.array(D, d, D)
            second term of gradient,
            ordered left-mid-right.
    """
    
    # calculate fixed points if not supplied
    if l is None or r is None:
        l, r = fixedPoints(A)
        
    # calculate first contraction
    term1 = ncon((l, r, A, A, np.conj(A), hTilde), ([-1, 1], [5, 7], [1, 3, 2], [2, 4, 5], [-3, 6, 7], [3, 4, -2, 6]))
    
    # calculate second contraction
    term2 = ncon((l, r, A, A, np.conj(A), hTilde), ([6, 1], [5, -3], [1, 3, 2], [2, 4, 5], [6, 7, -1], [3, 4, 7, -2]))
    
    return term1, term2


def reducedHamUniform(h, A, l=None, r=None):
    """
    Regularize Hamiltonian such that its expectation value is 0.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian that needs to be reduced,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        A : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
            
        Returns
        -------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight.
    """
    
    d = A.shape[1]
    
    # calculate fixed points if not supplied
    if l is None or r is None:
        l, r = fixedPoints(A)
    
    # calculate expectation value
    e = np.real(expVal2Uniform(h, A, l, r))
    
    # substract from hamiltonian
    hTilde = h - e * ncon((np.eye(d), np.eye(d)), ([-1, -3], [-2, -4]))
    
    return hTilde


def EtildeRight(A, l, r, v):
    """
    Implement the action of (1 - Etilde) on a right vector v.
    
        Parameters
        ----------
        A : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        v : np.array(D**2)
            right matrix of size (D, D) on which
            (1 - Etilde) acts,
            given as a vector of size (D**2,)
        
        Returns
        -------
        vNew : np.array(D**2)
            result of action of (1 - Etilde)
            on a right matrix,
            given as a vector of size (D**2,)
    """
    
    D = A.shape[0]
    
    # reshape to matrix
    v = v.reshape(D, D)
        
    # transfermatrix contribution
    transfer = ncon((A, np.conj(A), v), ([-1, 2, 1], [-2, 2, 3], [1, 3]))

    # fixed point contribution
    fixed = np.trace(l @ v) * r

    # sum these with the contribution of the identity
    vNew = v - transfer + fixed

    return vNew.reshape((D ** 2))


def RhUniform(hTilde, A, l=None, r=None):
    """
    Find the partial contraction for Rh.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        A : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        
        Returns
        -------
        Rh : np.array(D, D)
            result of contraction,
            ordered top-bottom.
    """
    
    D = A.shape[0]
    
    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)
    
    # construct b, which is the matrix to the right of (1 - E)^P in the figure above
    b = ncon((r, A, A, np.conj(A), np.conj(A), hTilde), ([4, 5], [-1, 2, 1], [1, 3, 4], [-2, 8, 7], [7, 6, 5], [2, 3, 8, 6]))
    
    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeRight, A, l, r))
    Rh = gmres(A, b.reshape(D ** 2))[0]
    
    return Rh.reshape((D, D))


def gradLeftTerms(hTilde, A, l=None, r=None):
    """
    Calculate the value of the left terms.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        
        Returns
        -------
        leftTerms : np.array(D, d, D)
            left terms of gradient,
            ordered left-mid-right.
    """
    
    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)
    
    # calculate partial contraction
    Rh = RhUniform(hTilde, A, l, r)
    
    # calculate full contraction
    leftTerms = ncon((Rh, A, l), ([1, -3], [2, -2, 1], [-1, 2]))
    
    return leftTerms


def EtildeLeft(A, l, r, v):
    """
    Implement the action of (1 - Etilde) on a left vector matrix v.
    
        Parameters
        ----------
        A : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        v : np.array(D**2)
            right matrix of size (D, D) on which
            (1 - Etilde) acts,
            given as a vector of size (D**2,)
        
        Returns
        -------
        vNew : np.array(D**2)
            result of action of (1 - Etilde)
            on a left matrix,
            given as a vector of size (D**2,)
    """
    
    D = A.shape[0]
    
    # reshape to matrix
    v = v.reshape(D, D)

    # transfer matrix contribution
    transfer = ncon((v, A, np.conj(A)), ([3, 1], [1, 2, -2], [3, 2, -1]))

    # fixed point contribution
    fixed = np.trace(v @ r) * l

    # sum these with the contribution of the identity
    vNew = v - transfer + fixed

    return vNew.reshape((D ** 2))


def LhUniform(hTilde, A, l=None, r=None):
    """
    Find the partial contraction for Lh.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        
        Returns
        -------
        Lh : np.array(D, D)
            result of contraction,
            ordered bottom-top.
    """
    
    D = A.shape[0]
    
    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)
    
    # construct b, which is the matrix to the right of (1 - E)^P in the figure above
    b = ncon((l, A, A, np.conj(A), np.conj(A), hTilde), ([5, 1], [1, 3, 2], [2, 4, -2], [5, 6, 7], [7, 8, -1], [3, 4, 6, 8]))    
    
    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeLeft, A, l, r)) 
    Lh = gmres(A, b.reshape(D ** 2))[0]
    
    return Lh.reshape((D, D))


def gradRightTerms(hTilde, A, l=None, r=None):
    """
    Calculate the value of the right terms.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        
        Returns
        -------
        rightTerms : np.array(D, d, D)
            right terms of gradient,
            ordered left-mid-right.
    """
    
    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)
    
    # calculate partial contraction
    Lh = LhUniform(hTilde, A, l, r)
    
    # calculate full contraction
    rightTerms = ncon((Lh, A, r), ([-1, 1], [1, -2, 2], [2, -3]))
    
    return rightTerms


def gradient(h, A, l=None, r=None):
    """
    Calculate the gradient of the expectation value of h @ MPS A.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalized.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalized.
        
        Returns
        -------
        grad : np.array(D, d, D)
            Gradient,
            ordered left-mid-right.
    """
    
    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)
        
    # renormalize Hamiltonian
    hTilde = reducedHamUniform(h, A, l, r)
        
    # find terms
    centerTerm1, centerTerm2 = gradCenterTerms(hTilde, A, l, r)
    leftTerms = gradLeftTerms(hTilde, A, l, r)
    rightTerms = gradRightTerms(hTilde, A, l, r)
    
    grad = 2 * (centerTerm1 + centerTerm2 + leftTerms + rightTerms)
    
    return grad


def groundStateGradDescent(h, D, eps=1e-1, A0=None, tol=1e-4, maxIter=1e4):
    """
    Find the ground state using gradient descent.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian to minimize,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        D : int
            Bond dimension
        eps : float
            Stepsize.
        A0 : np.array (D, d, D)
            normalized MPS tensor with 3 legs,
            ordered left-bottom-right,
            initial guess.
        tol : float
            Tolerance for convergence criterium.
        
        Returns
        -------
        E : float
            expectation value @ minimum
        A : np.array(D, d, D)
            ground state MPS,
            ordered left-mid-right.
    """
    
    d = h.shape[0]
    
    # if no initial value, choose random
    if A0 is None:
        A0 = createMPS(D, d)
        A0 = normalizeMPS(A0)
    
    # calculate gradient
    g = gradient(h, A0)
    
    A = A0
    
    i = 0
    while not(np.all(np.abs(g) < tol)):
        # do a step
        A = A - eps * g
        A = normalizeMPS(A)
        i += 1
        
        if not(i % 100):
            E = np.real(expVal2Uniform(h, A))
            print('Current energy:', E)
        
        # calculate new gradient
        g = gradient(h, A)
        
        if i > maxIter:
            print('Warning: gradient descent did not converge!')
            break
    
    # calculate ground state energy
    E = np.real(expVal2Uniform(h, A))
    
    return E, A


def groundStateMinimize(h, D, A0=None, tol=1e-4):
    """
    Find the ground state using a scipy minimizer.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian to minimize,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        D : int
            Bond dimension
        A0 : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            initial guess.
        tol : float
            Relative convergence criterium.
        
        Returns
        -------
        E : float
            expectation value @ minimum
        A : np.array(D, d, D)
            ground state MPS,
            ordered left-mid-right.
    """
    
    d = h.shape[0]
    
    def unwrapper(varA):
        """
        Unwraps real MPS vector to complex MPS tensor.
        
            Parameters
            ----------
            varA : np.array(2 * D * d * D)
                MPS tensor in real vector form.
            D : int
                Bond dimension.
            d : int
                Physical dimension.
                
            Returns
            -------
            A : np.array(D, d, D)
                MPS tensor with 3 legs,
                ordered left-bottom-right.
        """
        
        # unpack real and imaginary part
        Areal = varA[:D ** 2 * d]
        Aimag = varA[D ** 2 * d:]
        
        A = Areal + 1.0j * Aimag
        
        return np.reshape(A, (D, d, D))
    
    def wrapper(A):
        """
        Wraps MPS tensor to real MPS vector.
        
            Parameters
            ----------
            A : np.array(D, d, D)
                MPS tensor,
                ordered left-bottom-right
            
            Returns
            -------
            varA : np.array(2 * D * d * D)
                MPS tensor in real vector form.
        """
        
        # split into real and imaginary part
        Areal = np.real(A)
        Aimag = np.imag(A)
        
        # combine into vector
        varA = np.concatenate( (Areal.reshape(-1), Aimag.reshape(-1)) )
        
        return varA
    
    # if no initial MPS, take random one
    if A0 is None:
        A0 = createMPS(D, d)
        A0 = normalizeMPS(A0)
    
    # define f for minimize in scipy
    def f(varA):
        """
        Function to optimize via minimize.
        
            Parameters
            ----------
            varA : np.array(2 * D * d * D)
                MPS tensor in real vector form.
            
            Returns
            -------
            e : float
                function value @varA
            g : np.array(2 * D * d * D)
                gradient vector @varA
        """
        
        # unwrap varA
        A = unwrapper(varA)
        A = normalizeMPS(A)
        
        # calculate fixed points
        l, r = fixedPoints(A)
        
        # calculate function value and gradient
        e = np.real(expVal2Uniform(h, A, l, r))
        g = gradient(h, A, l, r)
        
        # wrap g
        g = wrapper(g)
        
        return e, g
    
    # calculate minimum
    result = minimize(f, wrapper(A0), jac=True, tol=tol)
    
    # unpack result
    E = result.fun
    A = unwrapper(result.x)
    
    return E, A


def Heisenberg(Jx, Jy, Jz, hz):
    """
    Construct the spin-1 Heisenberg Hamiltonian for given couplings.
    
        Parameters
        ----------
        Jx : float
            Coupling strength in x direction
        Jy : float
            Coupling strength in y direction
        Jy : float
            Coupling strength in z direction
        hz : float
            Coupling for Sz terms

        Returns
        -------
        h : np.array (3, 3, 3, 3)
            Spin-1 Heisenberg Hamiltonian.
    """
    Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    Sy = np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]]) * 1.0j /np.sqrt(2)
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    I = np.eye(3)

    return -Jx*ncon((Sx, Sx), ([-1, -3], [-2, -4]))-Jy*ncon((Sy, Sy), ([-1, -3], [-2, -4]))-Jz*ncon((Sz, Sz), ([-1, -3], [-2, -4]))             - hz*ncon((I, Sz), ([-1, -3], [-2, -4])) - hz*ncon((Sz, I), ([-1, -3], [-2, -4]))


def reducedHamMixed(h, Ac, Ar):
    """
    Regularize Hamiltonian such that its expectation value is 0.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian that needs to be reduced,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauged.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right gauged.

        Returns
        -------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight.
    """
    
    d = Ac.shape[1]
    
    # calculate expectation value
    e = np.real(expVal2Mixed(h, Ac, Ar))
    
    # substract from hamiltonian
    hTilde = h - e * ncon((np.eye(d), np.eye(d)), ([-1, -3], [-2, -4]))
    
    return hTilde


def RhMixed(hTilde, Ar, C, tol=1e-5):
    """
    Calculate Rh, for a given MPS in mixed gauge.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        tol : float, optional
            tolerance for gmres
            
        Returns
        -------
        Rh : np.array(D, D)
            result of contraction,
            ordered top-bottom.
    """
    
    D = Ar.shape[0]
    tol = max(tol, 1e-14)
    
    # construct fixed points for Ar
    l = np.conj(C).T @ C # left fixed point of right transfer matrix
    r = np.eye(D) # right fixed point of right transfer matrix: right orthonormal

    # construct b
    b = ncon((Ar, Ar, np.conj(Ar), np.conj(Ar), hTilde), ([-1, 2, 1], [1, 3, 4], [-2, 7, 6], [6, 5, 4], [2, 3, 7, 5]))
    
    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeRight, Ar, l, r))
    Rh = gmres(A, b.reshape(D ** 2), tol=tol)[0]
    
    return Rh.reshape((D, D))


def LhMixed(hTilde, Al, C, tol=1e-5):
    """
    Calculate Lh, for a given MPS in mixed gauge.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        tol : float, optional
            tolerance for gmres
            
        Returns
        -------
        Lh : np.array(D, D)
            result of contraction,
            ordered bottom-top.
    
    """
    
    D = Al.shape[0]
    tol = max(tol, 1e-14)
    
    # construct fixed points for Al
    l = np.eye(D) # left fixed point of left transfer matrix: left orthonormal
    r = C @ np.conj(C).T # right fixed point of left transfer matrix
        
    # construct b
    b = ncon((Al, Al, np.conj(Al), np.conj(Al), hTilde), ([4, 2, 1], [1, 3, -2], [4, 5, 6], [6, 7, -1], [2, 3, 5, 7]))
    
    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeLeft, Al, l, r)) 
    Lh = gmres(A, b.reshape(D ** 2), tol=tol)[0]
    
    return Lh.reshape((D, D))


def H_Ac(hTilde, Al, Ar, Lh, Rh, v):
    """
    Action of the effective Hamiltonian for Ac (131) on a vector.

        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.
        v : np.array(D, d, D)
            Tensor of size (D, d, D)

        Returns
        -------
        H_AcV : np.array(D, d, D)
            Result of the action of H_Ac on the vector v,
            representing a tensor of size (D, d, D)

    """

    # first term
    term1 = ncon((Al, v, np.conj(Al), hTilde), ([4, 2, 1], [1, 3, -3], [4, 5, -1], [2, 3, 5, -2]))

    # second term
    term2 = ncon((v, Ar, np.conj(Ar), hTilde), ([-1, 2, 1], [1, 3, 4], [-3, 5, 4], [2, 3, -2, 5]))

    # third term
    term3 = ncon((Lh, v), ([-1, 1], [1, -2, -3]))

    # fourth term
    term4 = ncon((v, Rh), ([-1, -2, 1], [1, -3]))

    # sum
    H_AcV = term1 + term2 + term3 + term4

    return H_AcV


def H_C(hTilde, Al, Ar, Lh, Rh, v):
    """
    Action of the effective Hamiltonian for Ac (131) on a vector.

        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.
        v : np.array(D, D)
            Matrix of size (D, D)

        Returns
        -------
        H_CV : np.array(D, D)
            Result of the action of H_C on the matrix v.

    """

    # first term
    term1 = ncon((Al, v, Ar, np.conj(Al), np.conj(Ar), hTilde), ([5, 3, 1], [1, 2], [2, 4, 7], [5, 6, -1], [-2, 8, 7], [3, 4, 6, 8]))

    # second term
    term2 = Lh @ v

    # third term
    term3 = v @ Rh

    # sum
    H_CV = term1 + term2 + term3

    return H_CV


def calcNewCenter(hTilde, Al, Ac, Ar, C, Lh=None, Rh=None, tol=1e-5):
    """
    Find new guess for Ac and C as fixed points of the maps H_Ac and H_C.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            diagonal.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.
        tol : float, optional
            current tolerance
            
        Returns
        -------
        AcTilde : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        CTilde : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
    """
    
    D = Al.shape[0]
    d = Al.shape[1]
    tol = max(tol, 1e-14)
    
    # calculate left en right environment if they are not given
    if Lh is None:
        Lh = LhMixed(hTilde, Al, C, tol)
    if Rh is None:
        Rh = RhMixed(hTilde, Ar, C, tol)
    
    # calculate new AcTilde
    
    # wrapper around H_Ac that takes and returns a vector
    handleAc = lambda v: (H_Ac(hTilde, Al, Ar, Lh, Rh, v.reshape(D, d, D))).reshape(-1)
    # cast to linear operator
    handleAcLO = LinearOperator((D ** 2 * d, D ** 2 * d), matvec=handleAc)
    # compute eigenvector
    _, AcTilde = eigs(handleAcLO, k=1, which="SR", v0=Ac.reshape(-1), tol=tol)
    
    
    # calculate new CTilde
    
    # wrapper around H_C that takes and returns a vector
    handleC = lambda v: (H_C(hTilde, Al, Ar, Lh, Rh, v.reshape(D, D))).reshape(-1)
    # cast to linear operator
    handleCLO = LinearOperator((D ** 2, D ** 2), matvec=handleC)
    # compute eigenvector
    _, CTilde = eigs(handleCLO, k=1, which="SR", v0=C.reshape(-1), tol=tol)
    
    # reshape to tensors of correct size
    AcTilde = AcTilde.reshape((D, d, D))
    CTilde = CTilde.reshape((D, D))
    
    return AcTilde, CTilde


def minAcC(AcTilde, CTilde, tol=1e-5):
    """
    Find Al and Ar corresponding to Ac and C, according to algorithm 5 in the lecture notes.
    
        Parameters
        ----------
        AcTilde : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            new guess for center gauge. 
        CTilde : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            new guess for center gauge
        
        Returns
        -------
        Al : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge. 
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right,
            center gauge
    
    """
    
    D = AcTilde.shape[0]
    d = AcTilde.shape[1]
    tol = max(tol, 1e-14)
    
    # polar decomposition of Ac
    UlAc, _ = polar(AcTilde.reshape((D * d, D)))
                    
    # polar decomposition of C
    UlC, _ = polar(CTilde)
    
    # construct Al
    Al = (UlAc @ np.conj(UlC).T).reshape(D, d, D)
    
    # find corresponding Ar, C, and Ac through right orthonormalizing Al
    C, Ar = rightOrthonormalize(Al, CTilde, tol=tol)
    nrm = np.trace(C @ np.conj(C).T)
    C = C / np.sqrt(nrm)
    Ac = ncon((Al, C), ([-1, -2, 1], [1, -3]))
    
    return Al, Ac, Ar, C


def gradientNorm(hTilde, Al, Ac, Ar, C, Lh, Rh):
    """
    Calculate the norm of the gradient.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalized.
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        Lh : np.array(D, D)
            left environment,
            ordered bottom-top.
        Rh : np.array(D, D)
            right environment,
            ordered top-bottom.
        
        Returns
        -------
        norm : float
            norm of the gradient @Al, Ac, Ar, C
    """
        
    # calculate update on Ac and C using maps H_Ac and H_c
    AcUpdate = H_Ac(hTilde, Al, Ar, Lh, Rh, Ac)
    CUpdate = H_C(hTilde, Al, Ar, Lh, Rh, C)
    AlCupdate = ncon((Al, CUpdate), ([-1, -2, 1], [1, -3]))
    
    norm = np.linalg.norm(AcUpdate - AlCupdate)
    
    return norm


def vumps(h, D, A0=None, tol=1e-4, tolFactor=1e-2, verbose=True):
    """
    Find the ground state of a given Hamiltonian using VUMPS.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian to minimize,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        D : int
            Bond dimension
        A0 : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            initial guess.
        tol : float
            Relative convergence criterium.
        
        Returns
        -------
        E : float
            expectation value @ minimum
        A : np.array(D, d, D)
            ground state MPS,
            ordered left-mid-right.
    """
    
    d = h.shape[0]
    
    # if no initial guess, random one
    if A0 is None:
        A0 = createMPS(D, d)
    
    # go to mixed gauge
    Al, Ac, Ar, C = mixedCanonical(A0)
    
    flag = True
    delta = 1e-5
    
    while flag:
        # regularize H
        hTilde = reducedHamMixed(h, Ac, Ar)
        
        # calculate environments
        Lh = LhMixed(hTilde, Al, C, tol=delta*tolFactor)
        Rh = RhMixed(hTilde, Ar, C, tol=delta*tolFactor)
        
        # calculate new center
        AcTilde, CTilde = calcNewCenter(hTilde, Al, Ac, Ar, C, Lh, Rh, tol=delta*tolFactor)
        
        # find Al, Ar from Ac, C
        AlTilde, AcTilde, ArTilde, CTilde = minAcC(AcTilde, CTilde, tol=delta*tolFactor**2)
        
        # calculate norm
        delta = gradientNorm(hTilde, Al, Ac, Ar, C, Lh, Rh)
        
        # check convergence
        if delta < tol:
            flag = False
        
        # update tensors
        Al, Ac, Ar, C = AlTilde, AcTilde, ArTilde, CTilde
        
        # print current energy, optional...
        E = np.real(expVal2Mixed(h, Ac, Ar))
        print('Current energy:', E)
    
    return E, Al, Ac, Ar, C


"""
Chapter 3
"""

def leftFixedPointMPO(O, Al, tol):
    """
    Computes the left fixed point (250).

        Parameters
        ----------
        O : np.array (d, d, d, d)
            MPO tensor,
            ordered left-top-right-bottom.
        Al : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left-orthonormal.
        tol : float, optional
            current tolerance

        Returns
        -------
        lam : float
            Leading left eigenvalue.
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.

    """
    D = Al.shape[0]
    d = Al.shape[1]
    
    # construct handle for the action of the relevant operator and cast to linear operator
    transferLeftHandleMPO = lambda v: (ncon((v.reshape((D,d,D)), Al, np.conj(Al), O),([5, 3, 1], [1, 2, -3], [5, 4, -1], [3, 2, -2, 4]))).reshape(-1)
    transferLeftMPO = LinearOperator((D**2*d, D**2*d), matvec=transferLeftHandleMPO)
    lam, Fl = eigs(transferLeftMPO, k=1, which="LM", tol=tol)
    
    return lam, Fl.reshape((D,d,D))


def rightFixedPointMPO(O, Ar, tol):
    """
    Computes the right fixed point (250).

        Parameters
        ----------
        O : np.array (d, d, d, d)
            MPO tensor,
            ordered left-top-right-bottom.
        Ar : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right-orthonormal.
        tol : float, optional
            current tolerance

        Returns
        -------
        lam : float
            Leading right eigenvalue.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.

    """
    D = Ar.shape[0]
    d = Ar.shape[1]
    
    # construct handle for the action of the relevant operator and cast to linear operator
    transferRightHandleMPO = lambda v: (ncon((v.reshape(D, d, D), Ar, np.conj(Ar), O), ([1, 3, 5], [-1, 2, 1], [-3, 4, 5], [-2, 2, 3, 4]))).reshape(-1)
    transferRightMPO = LinearOperator((D**2*d, D**2*d), matvec=transferRightHandleMPO)
    lam, Fr = eigs(transferRightMPO, k=1, which="LM", tol=tol)
    
    return lam, Fr.reshape((D,d,D))


def overlapFixedPointsMPO(Fl, Fr, C):
    """
    Performs the contraction that gives the overlap of the fixed points (251).

        Parameters
        ----------
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.

        Returns
        -------
        overlap : float
            Overlap of the fixed points.

    """
    
    overlap = ncon((Fl, Fr, C, np.conj(C)), ([1, 3, 2], [5, 3, 4], [2, 5], [1, 4]))
    
    return overlap


def O_Ac(X, O, Fl, Fr, lam):
    """
    Action of the map (256) on a given tensor.

        Parameters
        ----------
        X : np.array(D, d, D)
            Tensor of size (D, d, D)
        O : np.array (d, d, d, d)
            MPO tensor,
            ordered left-top-right-bottom.
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.
        lam : float
            Leading eigenvalue.

        Returns
        -------
        Xnew : np.array(D, d, D)
            Result of the action of O_Ac on the tensor X.

    """
    
    Xnew = ncon((Fl, Fr, X, O),([-1, 2, 1], [4, 5, -3], [1, 3, 4], [2, 3, 5, -2])) / lam
    
    return Xnew


def O_C(X, Fl, Fr):
    """
    Action of the map (257) on a given tensor.

        Parameters
        ----------
        X : np.array(D, D)
            Tensor of size (D, D)
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.

        Returns
        -------
        Xnew : np.array(D, d, D)
            Result of the action of O_C on the tensor X.

    """
    
    Xnew = ncon((Fl, Fr, X), ([-1, 3, 1], [2, 3, -2], [1, 2]))
    
    return Xnew


def calcNewCenterMPO(O, Ac, C, Fl, Fr, lam, tol=1e-5):
    """
    Find new guess for Ac and C as fixed points of the maps O_Ac and O_C.
    
        Parameters
        ----------
        O : np.array (d, d, d, d)
            MPO tensor,
            ordered left-top-right-bottom.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.
        lam : float
            Leading eigenvalue.
        tol : float, optional
            current tolerance
    
        Returns
        -------
        AcTilde : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        CTilde : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.

    """
    
    D = Ac.shape[0]
    d = Ac.shape[1]
    
    # construct handle for O_Ac map and cast to linear operator
    handleAc = lambda X: (O_Ac(X.reshape((D,d,D)), O, Fl, Fr, lam)).reshape(-1)
    handleAc = LinearOperator((D**2*d, D**2*d), matvec=handleAc)
    # construct handle for O_C map and cast to linear operator
    handleC = lambda X: (O_C(X.reshape(D, D), Fl, Fr)).reshape(-1)
    handleC = LinearOperator((D**2, D**2), matvec=handleC)
    # compute fixed points of these maps: gives new guess for center tensors
    _, AcTilde = eigs(handleAc, k=1, which="LM", v0=Ac.reshape(-1), tol=tol)
    _, CTilde = eigs(handleC, k=1, which="LM", v0=C.reshape(-1), tol=tol)
    
    # reshape to tensors of correct size
    AcTilde = AcTilde.reshape((D, d, D))
    CTilde = CTilde.reshape((D, D))
    
    return AcTilde, CTilde


def vumpsMpo(O, D, A0=None, tol=1e-4):
    """
    Find the fixed point MPS of a given MPO using VUMPS.
    
        Parameters
        ----------
        O : np.array (d, d, d, d)
            MPO tensor,
            ordered left-top-right-bottom.
        D : int
            Bond dimension
        A0 : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            initial guess.
        tol : float
            Relative convergence criterium.
        
        Returns
        -------
        lam : float
            Leading eigenvalue.
        Al : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right,
            center gauge.
        C : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.
        
    """
    
    d = O.shape[0]
    
    # if no initial guess, random one
    if A0 is None:
        A0 = createMPS(D, d)
    
    # go to mixed gauge
    Al, Ac, Ar, C = mixedCanonical(A0)
    
    delta = 1e-4
    flag = True
    while flag:
        # compute left and right fixed points
        lam, Fl = leftFixedPointMPO(O, Al, delta/10)
        _ , Fr = rightFixedPointMPO(O, Ar, delta/10)
        Fl /= overlapFixedPointsMPO(Fl, Fr, C)
        lam = np.real(lam)[0]
        # compute updates on Ac and C
        AcTilde, CTilde = calcNewCenterMPO(O, Ac, C, Fl, Fr, lam, delta/10)
        AlTilde, AcTilde, ArTilde, CTilde = minAcC(AcTilde, CTilde)
        # calculate convergence measure, check for convergence
        delta = np.linalg.norm(O_Ac(Ac, O, Fl, Fr, lam) - ncon((Al, O_C(C, Fl, Fr)), ([-1, -2, 1], [1, -3])))
        if delta < tol:
            flag = False
        # update tensors
        Al, Ac, Ar, C = AlTilde, AcTilde, ArTilde, CTilde
    
    return lam, Al, Ac, Ar, C, Fl, Fr


def isingO(beta, J):
    """
    Gives the MPO tensor corresponding to the partition function of the 2d 
    classical Ising model at a given temperature and coupling, obtained by
    distributing the Boltzmann weights evenly over all vertices.
    
        Parameters
        ----------
        beta : float
            Inverse temperature.
        J : float
            Coupling strength.
    
        Returns
        -------
        O : np.array (2, 2, 2, 2)
            MPO tensor,
            ordered left-top-right-bottom.

    """
    # basic vertex tensor
    vertex = np.zeros( (2,) * 4 )
    vertex[tuple([np.arange(2)] * 4)] = 1
    # build square root of matrix of Boltzmann weights and pull into vertex edges
    c, s = np.sqrt(np.cosh(beta*J)), np.sqrt(np.sinh(beta*J))
    Qsqrt = 1/np.sqrt(2) * np.array([[c+s, c-s],[c-s, c+s]])
    O = ncon((Qsqrt, Qsqrt, Qsqrt, Qsqrt, vertex), ([-1,1], [-2,2], [-3,3], [-4,4], [1,2,3,4]))
    return O

def isingM(beta, J):
    """
    Gives the magnetizatopn MPO tensor for the 2d classical Ising model at a
    given temperature and coupling.
    
        Parameters
        ----------
        beta : float
            Inverse temperature.
        J : float
            Coupling strength.
    
        Returns
        -------
        M : np.array (2, 2, 2, 2)
            Magnetization MPO tensor,
            ordered left-top-right-bottom.

    """
    vertex = np.zeros( (2,) * 4 )
    vertex[tuple([np.arange(2)] * 4)] = 1
    Z = np.array([[1,0],[0,-1]])
    c, s = np.sqrt(np.cosh(beta*J)), np.sqrt(np.sinh(beta*J))
    Qsqrt = 1/np.sqrt(2) * np.array([[c+s, c-s],[c-s, c+s]])
    vertexZ = ncon((Z, vertex), ([-1,1], [1,-2,-3,-4]))
    M = ncon((Qsqrt, Qsqrt, Qsqrt, Qsqrt, vertexZ), ([-1,1], [-2,2], [-3,3], [-4,4], [1,2,3,4]))
    return M


def isingMagnetization(beta, J, Ac, Fl, Fr):
    """
    Computes the expectation value of the magnetization in the Ising model
    for a given temperature and coupling
    
        Parameters
        ----------
        beta : float
            Inverse temperature.
        J : float
            Coupling strength.
        Ac : np.array(D, d, D)
            MPS tensor of the MPS fixed point,
            with 3 legs ordered left-bottom-right,
            center gauge.
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.
    
        Returns
        -------
        M : float
            Expectation value of the magnetization at the given temperature
            and coupling.

    """
    return ncon((Fl, Ac, isingM(beta, J), np.conj(Ac), Fr), (
        [1, 3, 2], [2,7,5],[3,7,8,6],[1,6,4], [5,8,4]))


def isingZ(beta, J, Ac, Fl, Fr):
    """
    Computes the Ising model partition function for a given temperature and
    coupling
    
        Parameters
        ----------
        beta : float
            Inverse temperature.
        J : float
            Coupling strength.
        Ac : np.array(D, d, D)
            MPS tensor of the MPS fixed point,
            with 3 legs ordered left-bottom-right,
            center gauge.
        Fl : np.array(D, d, D)
            left fixed point,
            ordered bottom-middle-top.
        Fr : np.array(D, d, D)
            right fixed point,
            ordered top-middle-bottom.
    
        Returns
        -------
        Z : float
            Value of the partition function at the given temperature and
            coupling.

    """
    
    Z = ncon((Fl, Ac, isingO(beta, J), np.conj(Ac), Fr), (
        [1, 3, 2], [2,7,5],[3,7,8,6],[1,6,4], [5,8,4]))
    
    return Z

def isingExact(beta, J):
    """
    Exact Onsager solution for the 2d classical Ising Model

        Parameters
        ----------
        beta : float
            Inverse temperature.
        J : float
            Coupling strength.
    
        Returns
        -------
        magnetization : float
            Magnetization at given temperature and coupling.
        free : float
            Free energy at given temperature and coupling.
        energy : float
            Energy at given temperature and coupling.

    """
    theta = np.arange(0, np.pi/2, 1e-6)
    x = 2 * np.sinh(2 * J * beta) / np.cosh(2 * J * beta) ** 2
    if 1 - (np.sinh(2 * J * beta)) ** (-4) > 0:
        magnetization = (1 - (np.sinh(2 * J * beta)) ** (-4)) ** (1 / 8)
    else:
        magnetization = 0
    free = -1 / beta * (np.log(2 * np.cosh(2 * J * beta)) + 1 / np.pi * np.trapz(np.log(1 / 2 * (1 + np.sqrt(1 - x ** 2 * np.sin(theta) ** 2))), theta))
    K = np.trapz(1 / np.sqrt(1 - x ** 2 * np.sin(theta) ** 2), theta)
    energy = -J * np.cosh(2 * J * beta) / np.sinh(2 * J * beta) * (1 + 2 / np.pi * (2 * np.tanh(2 * J * beta) ** 2 - 1) * K)
    return magnetization, free, energy
