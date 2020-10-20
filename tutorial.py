import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import svd, qr, rq

# choose bond dimension
D = 2

def createMPS(bondDimension, physDimension):
    # function to create a random MPS tensor for some bondDimension and physical dimension.
    # returns a 3-legged tensor (leftLeg - physLeg - rightLeg)
    
    return np.random.rand(bondDimension, physDimension, bondDimension) \
        + 1j*np.random.rand(bondDimension, physDimension, bondDimension)
    
def createTransfer(A):
    # function to return a transfer matrix starting from a given MPS tensor.
    # returns a 4-legged tensor (topLeft - bottomLeft - topRight - bottomRight)
    
    return np.einsum('isk,jsl->ijkl', A, np.conj(A))

# left and right fixed points, left and right orthonormal form, mixed gauge form

def leftFixedPoint(A):
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
    
    return lam, np.resize(rhoL,(D, D)) # hier mogelijks nog transpose nodig !!

def rightFixedPoint(A):
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
    
    return lam, np.resize(rhoR,(D, D))

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

# test code for functions listed above
D = 2
d = 3


A = createMPS(D, d)
T = createTransfer(A)
print(T.shape)
lamR, rhoR = rightFixedPoint(A)
lamL, rhoL = leftFixedPoint(A)

# check if eigenvalues are the same
if abs(lamR - lamL) < 0.01:
    print("left and right eigenvalues are the same:")
    print(lamR)
else:
    print("ERROR: different eigenvalues!")

print(T.shape)
print(rhoR)
# check if fixed points are correct
LHS = np.einsum('ijkl,kl->ij', T, rhoR)
RHS = lamR*rhoR
if np.allclose(LHS, RHS):
    print("right fixed point equation is OK")
else:
    print("ERROR: failed right fixed point check")

LHS = np.einsum('ijkl,ij->kl', T, rhoL)
RHS = lamL*rhoL
if np.allclose(LHS, RHS):
    print("left fixed point equation is OK")
else:
    print("ERROR: failed left fixed point check")
    
def QRPositive(A):
    # function that implements a QR decomposition of a matrix A, such
    # that the diagonal elements of R are positive, R is upper triangular and
    # Q is an isometry with A = QR
    # returns (Q, R)
    
    # QR decomposition
    Q, R = np.linalg.qr(A)
    print('before')
    print(Q)
    print(R)
    
    
    # extract signs and multiply
    D = np.diag(R)
#    D.setflags(write=1)
#    C = np.where(np.abs(D) < 1e-10)
#    D[C] = 1
    D = np.sign(D)
    Q, R = np.multiply(Q, D.T), np.multiply(D, R)
    print('after')
    print(Q)
    print(R)
#    diagSigns = np.sign(np.diag(R))
    return Q, R

def leftOrthonormal(A):
    # function that brings MPS A into left orthonormal gauge, such that
    # L * A = A_L * L
    # returns (L, A_L)
    
    D = A.shape[0]
    d = A.shape[1]
    
    # random guess for L
    L = np.random.rand(D,D)+1j*np.random.rand(D,D)
    L_norm = np.linalg.norm(L)
    L /= L_norm
    L_old = L
    LA = np.einsum('ik,ksj->isj', L, A)
    A_L, L = QRPositive(np.resize(LA, (D*d, D)))
    Lambda = np.linalg.norm(L)
    L /= Lambda
    delta = np.linalg.norm(L-L_old)
    print(delta)
    
    # Decompose L*A until L converges
    while delta > 1e-10:
        L_old = L
        Lambda = np.linalg.norm(L)
        L /= Lambda
        LA = np.einsum('ik,ksj->isj', L, A)
        A_L, L = QRPositive(np.resize(LA, (D*d, D)))
        delta = np.linalg.norm(L-L_old)
        print(delta)
    return L, np.resize(A_L, (D,d,D)), np.linalg.norm(L)
leftOrthonormal(normaliseMPS(createMPS(2,3))[0])

def RQPositive(A):
    # function that implements a RQ decomposition of a matrix A, such that
    # the diagonal elements of R are positive, R is upper triangular and Q
    # is an isometry with A = RQ
    # returns (R, Q)
    
    # RQ decomposition
    R, Q = rq(A)
    
    # extract signs and multiply
    diagSigns = np.sign(np.diag(R))
    return R*diagSigns, diagSigns*Q

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
        AR = np.einsum('ijk,kl->ijl', A,R)
        Rnew, A_R = RQPositive(np.resize(AR, (D*d, D)))
        convergence = np.linalg.norm(Rnew-R)
        R = Rnew
    
    return R, np.resize(A_R, (D,d,D))


#very random test case to check entanglement entropy and truncation
D = 2
d = 3

#some random tensors to check if everything at least works
aL = np.random.rand(D,d,D)
aR = np.random.rand(D,d,D)
l = np.random.rand(D,D)
r = np.random.rand(D,D)

#normal calculation
c = entanglementSpectrum(aL,aR,l,r)
#print(c[0].shape, c[1].shape, c[2], c[3])

#calculate with truncation step
c = entanglementSpectrum(aL,aR,l,r,4)
#print(c[0].shape, c[1].shape, c[2], c[3])