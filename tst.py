from tutorialFunctions import *

"""numpy eigs provides efficient way to find largest eigenvalue/vector
    arg1 = matrix-like operator to find eigenvalue
    k = amount of eigenvalues/vectors to return
    which = 'LM' select largest magnitude eigenvalues"""

d = 3 # physical dimension
D = 5 # bond dimension

A = np.random.rand(D, d, D) + 1j*np.random.rand(D, d, D)

# E = np.einsum('isk,jsl->ijkl', A, np.conj(A))

# # right fixed point
# lambdaRight, r = eigs(np.resize(E, (D**2, D**2)), k=1, which='LM')
# r.resize(D,D)

# # left fixed point (via transpose transfer matrix)
# lambdaLeft, l = eigs(np.resize(E, (D**2, D**2)).T, k=1, which='LM')
# l = np.resize(l, (D,D)).T # note transpose!

# # normalise A
# A = A / np.sqrt(lambdaRight)

# # normalise transfer matrix
# E = np.einsum('isk,jsl->ijkl', A, np.conj(A))

# # normalise fixed points
# norm = np.sqrt(np.einsum('ij,ji->', l, r))
# l = l / norm
# r = r / norm

A, l, r = normaliseMPS(A)

# assert abs(lambdaLeft - lambdaRight) < 1e-12, "Left and right fixed point values should be the same!"
assert np.allclose(l, np.einsum('ijk,li,ljm->mk', A, l, np.conj(A)), 1e-12), "l should be a left fixed point!"
assert np.allclose(r, np.einsum('ijk,kl,mjl->im', A, r, np.conj(A)), 1e-12), "r should be a right fixed point!"
assert abs(np.einsum('ij,ji->', l, r)-1) < 1e-12, "Left and right fixed points should be trace normalised!"

Al, Ar, Ac, C = mixedCanonical(A, L0=None, R0=None, tol=1e-14, maxIter=1e5)

assert np.allclose(np.einsum('ijk,ijl->kl', Al, np.conj(Al)), np.eye(D)), "Al1 not in left-orthonormal form"
assert np.allclose(np.einsum('ijk,ljk->il', Ar, np.conj(Ar)), np.eye(D)), "Ar not in right-orthonormal form"
LHS = np.einsum('ijk,kl->ijl', Al, C)
RHS = np.einsum('ij,jkl->ikl', C, Ar)
assert np.allclose(LHS, RHS) and np.allclose(RHS/np.sqrt(np.einsum('ijk,ijk', RHS, np.conj(RHS))), Ac), "Something went wrong in gauging the MPS"

# set optimal contraction sequence
path = ['einsum_path', (0, 2), (0, 1)]

# calculate transfer matrix handle and cast to LinearOperator
transferLeftHandle = lambda v: np.reshape(
    np.einsum('ijk,ljm,li->mk', A, np.conj(A), v.reshape((D, D)), optimize=path), D ** 2)

v = np.random.rand(D**2) + 1j*np.random.rand(D**2)

print(leftHandle(A, v))
print(transferLeftHandle(v))

assert np.allclose(transferLeftHandle(v), leftHandle(A, v), 1e-12)
