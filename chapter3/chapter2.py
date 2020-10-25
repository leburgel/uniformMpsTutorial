"""
Summary of all functions used and defined in notebook chapter 2
"""

from chapter1 import *
from scipy.optimize import minimize
from scipy.sparse.linalg import gmres
from scipy.linalg import polar
from functools import partial


def gradCenterTerms(hTilde, A, l=None, r=None):
    """
    Calculate the value of the center terms.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        A : np.array (D, d, D)
            normalised MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
        
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
    Regularise Hamiltonian such that its expectation value is 0.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian that needs to be reduced,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        A : np.array (D, d, D)
            normalised MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
            
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
            normalised MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
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
            renormalised.
        A : np.array (D, d, D)
            normalised MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
        
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
            renormalised.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
        
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
            normalised MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
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
            renormalised.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
        
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
            renormalised.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
        
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
            renormalised.
        A : np.array (D, d, D)
            MPS tensor with 3 legs,
            ordered left-bottom-right.
        l : np.array(D, D), optional
            left fixed point of transfermatrix,
            normalised.
        r : np.array(D, D), optional
            right fixed point of transfermatrix,
            normalised.
        
        Returns
        -------
        grad : np.array(D, d, D)
            Gradient,
            ordered left-mid-right.
    """
    
    # if l, r not specified, find fixed points
    if l is None or r is None:
        l, r = fixedPoints(A)
        
    # renormalise Hamiltonian
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
            Hamiltonian to minimise,
            ordered topLeft-topRight-bottomLeft-bottomRight.
        D : int
            Bond dimension
        eps : float
            Stepsize.
        A0 : np.array (D, d, D)
            normalised MPS tensor with 3 legs,
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
        A0 = normaliseMPS(A0)
    
    # calculate gradient
    g = gradient(h, A0)
    g0 = np.zeros((D, d, D))
    
    A = A0
    
    i = 0
    while not(np.all(np.abs(g) < tol)):
        # do a step
        A = A - eps * g
        A = normaliseMPS(A)
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


def groundStateMinimise(h, D, A0=None, tol=1e-4):
    """
    Find the ground state using a scipy minimizer.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian to minimise,
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
        A0 = normaliseMPS(A0)
    
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
        A = normaliseMPS(A)
        
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
    Regularise Hamiltonian such that its expectation value is 0.
    
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


def RhMixed(hTilde, Ar, C, tol=1e-3):
    """
    Calculate Rh, for a given MPS in mixed gauge.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalised.
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
    
    # construct fixed points for Ar
    l = np.conj(C).T @ C # left fixed point of right transfer matrix
    r = np.eye(D) # right fixed point of right transfer matrix: right orthonormal

    # construct b
    b = ncon((Ar, Ar, np.conj(Ar), np.conj(Ar), hTilde), ([-1, 2, 1], [1, 3, 4], [-2, 7, 6], [6, 5, 4], [2, 3, 7, 5]))
    
    # solve Ax = b for x
    A = LinearOperator((D ** 2, D ** 2), matvec=partial(EtildeRight, Ar, l, r))
    Rh = gmres(A, b.reshape(D ** 2), tol=tol)[0]
    
    return Rh.reshape((D, D))


def LhMixed(hTilde, Al, C, tol=1e-3):
    """
    Calculate Lh, for a given MPS in mixed gauge.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalised.
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
            renormalised.
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
            renormalised.
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


def calcNewCenter(hTilde, Al, Ac, Ar, C, Lh=None, Rh=None, tol=1e-3):
    """
    Find new guess for Ac and C as fixed points of the maps H_Ac and H_C.
    
        Parameters
        ----------
        hTilde : np.array (d, d, d, d)
            reduced Hamiltonian,
            ordered topLeft-topRight-bottomLeft-bottomRight,
            renormalised.
        Al : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor zith 3 legs,
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
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            center gauge.
        CTilde : np.array(D, D)
            Center gauge with 2 legs,
            ordered left-right.
    """
    
    D = Al.shape[0]
    d = Al.shape[1]
    
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


def minAcC(AcTilde, CTilde):
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
    
    # polar decomposition of Ac
    UlAc, _ = polar(AcTilde.reshape((D * d, D)))
                    
    # polar decomposition of C
    UlC, _ = polar(CTilde)
    
    # construct Al
    Al = (UlAc @ np.conj(UlC).T).reshape(D, d, D)
    
    # find corresponding Ar, C, and Ac through right orthonormalising Al
    C, Ar = rightOrthonormalise(Al)
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
            renormalised.
        Al : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            left orthonormal.
        Ar : np.array(D, d, D)
            MPS tensor zith 3 legs,
            ordered left-bottom-right,
            right orthonormal.
        Ac : np.array(D, d, D)
            MPS tensor zith 3 legs,
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
    
    D = Al.shape[0]
    d = Al.shape[1]
    
    # calculate update on Ac and C using maps H_Ac and H_c
    AcUpdate = H_Ac(hTilde, Al, Ar, Lh, Rh, Ac)
    CUpdate = H_C(hTilde, Al, Ar, Lh, Rh, C)
    AlCupdate = ncon((Al, CUpdate), ([-1, -2, 1], [1, -3]))
    
    norm = np.linalg.norm(AcUpdate - AlCupdate)
    
    return norm


def vumps(h, D, A0=None, tol=1e-4):
    """
    Find the ground state of a given Hamiltonian using VUMPS.
    
        Parameters
        ----------
        h : np.array (d, d, d, d)
            Hamiltonian to minimise,
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
        # regularise H
        hTilde = reducedHamMixed(h, Ac, Ar)
        
        # calculate environments
        Lh = LhMixed(hTilde, Al, C, tol=delta/10)
        Rh = RhMixed(hTilde, Ar, C, tol=delta/10)
        
        # calculate new center
        AcTilde, CTilde = calcNewCenter(hTilde, Al, Ac, Ar, C, Lh, Rh, tol=delta/10)
        
        # find Al, Ar from Ac, C
        AlTilde, AcTilde, ArTilde, CTilde = minAcC(AcTilde, CTilde)
        
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
