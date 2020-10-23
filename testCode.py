from tutorialFunctions import *
import matplotlib.pyplot as plt
from time import time
### A first test case for the gradient in python
D = 100
d = 3
J = -1

H = Heisenberg(J, J, J, 0)

A = createMPS(D, d)
ReA = np.real(A)
ImA = np.imag(A)

if False:
    # calculatee optimal paths for gradient contractions
    r = np.ones((D, D))
    l = np.ones((D, D))
    path = np.einsum_path('ijk,klm,jlqo,rqp,pon,ri,mn', A, A, H, np.conj(A), np.conj(A), l, r, optimize='optimal')

    print(path)

# extra haakjes om real(g) en imag(g) in tuple te plaatsen voor concate anders error !!!
varA = np.concatenate((ReA.reshape(-1), ImA.reshape(-1)))

print('Bond dimension: D =', D)

if False:
    # test gradient descent
    
    #pr.enable()
    EnergyHandle = partial(energyWrapper, H, D, d)
    t0 = time()
    res = minimize(EnergyHandle, varA, jac=True, tol=1e-4)
    print('Time for gradient descent optimization:', time()-t0, 's')
    Aopt = res.x
    print('Procedure converged at energy', res.fun, '\n')
    #pr.disable()
    # pr.print_stats()
    
if True:
    # test vumps
    
    tol = 1e-3
    A = normaliseMPS(createMPS(D, d))[0]
    Al, Ar, Ac, C = mixedCanonical(A)
    
    assert np.allclose(np.einsum('ijk,ijl->kl', Al, np.conj(Al)), np.eye(D)), "Al not in left-orthonormal form"
    assert np.allclose(np.einsum('ijk,ljk->il', Ar, np.conj(Ar)), np.eye(D)), "Ar not in right-orthonormal form"
    LHS = np.einsum('ijk,kl->ijl', Al, C)
    RHS = np.einsum('ij,jkl->ikl', C, Ar)
    assert np.allclose(LHS, RHS) and np.allclose(RHS/np.sqrt(np.einsum('ijk,ijk', RHS, np.conj(RHS))), Ac), "Something went wrong in gauging the MPS"
    
    flag = 1
    delta = 1e-4
    t0 = time()
    while flag:
        e = np.real(twoSiteMixed(H, Ac, Ar))
        print(e)
        hTilde = H - e * np.einsum("ik,jl->ijkl", np.eye(d), np.eye(d))
        Rh = rightEnvMixed(Ar, C, hTilde, delta)
        Lh = leftEnvMixed(Al, C, hTilde, delta)
        AcPrime, CPrime = calcNewCenter(Al, Ar, Ac, C, Lh, Rh, hTilde, delta)
        AlPrime, ArPrime, AcPrime, CPrime = minAcC(AcPrime, CPrime)
        delta = np.linalg.norm(H_Ac(Ac, Al, Ar, Rh, Lh, hTilde) - np.einsum('ijk,kl->ijl', Al, H_C(C, Al, Ar, Rh, Lh, hTilde)))
        # print(delta)
        Al = AlPrime; Ar = ArPrime; Ac = AcPrime; C = CPrime;
        if delta < tol:
            flag = 0
    print('Time for VUMPS optimization:', time()-t0, 's')
    print('Procedure converged at energy ', np.real(twoSiteMixed(H, Ac, Ar)), '\n')
    [_, S, _] = svd(C)
    plt.figure()
    plt.scatter(np.arange(D), S, marker='x')
    plt.show()

