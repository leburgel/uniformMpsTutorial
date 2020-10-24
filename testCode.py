from tutorialFunctions import *
from time import time
### A first test case for the gradient in python
D = 200
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
    # test Hamiltonian vumps
    D = 12
    d = 3
    J = -1
    H = Heisenberg(J, J, J, 0)

    print('Bond dimension: D =', D)
    
    tol = 1e-4
    A = normaliseMPS(createMPS(D, d))[0]
    Al, Ar, Ac, C = mixedCanonical(A)
    
    # check mixed canonical form
    assert np.allclose(np.einsum('ijk,ijl->kl', Al, np.conj(Al)), np.eye(D)), "Al not in left-orthonormal form"
    assert np.allclose(np.einsum('ijk,ljk->il', Ar, np.conj(Ar)), np.eye(D)), "Ar not in right-orthonormal form"
    LHS = np.einsum('ijk,kl->ijl', Al, C)
    RHS = np.einsum('ij,jkl->ikl', C, Ar)
    assert np.allclose(LHS, RHS) and np.allclose(RHS/np.sqrt(np.einsum('ijk,ijk', RHS, np.conj(RHS))), Ac), "Something went wrong in gauging the MPS"
    
    flag = 1
    delta = 1e-5
    i = 0
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
        i += 1
        if delta < tol:
            flag = 0
    print('Time for VUMPS optimization:', time()-t0, 's')
    print('Iterations needed:', i)
    print('Procedure converged at energy ', np.real(twoSiteMixed(H, Ac, Ar)), '\n')
    [_, S, _] = svd(C)
    plt.figure()
    plt.scatter(np.arange(D), S, marker='x')
    plt.yscale('log')
    plt.show()


if False:
    # test vumps for 2d ising partition function
    D = 12
    d = 2
    
    print('Bond dimension: D =', D)
    
    beta = .45 # critical point: 0.440686793509772

    A = createMPS(D, d)
    Al, Ar, Ac, C = mixedCanonical(A)
    O = isingO(beta, 1)
    delta = 1e-4
    tol = 1e-5
    flag = 1
    i = 0
    t0 = time()
    while flag:
        lam, Fl = leftFixedPointMPO(Al, O, delta)
        _ , Fr = rightFixedPointMPO(Ar, O, delta)
        Fl /= overlapFixedPointsMPO(Fl, Fr, C)
        lam = np.real(lam)[0]
        AcPrime, cPrime = calcNewCenterMPO(Ac, C, Fl, Fr, O, lam, delta)
        AlPrime, ArPrime, AcPrime, cPrime = minAcC(AcPrime, cPrime)
        delta = np.linalg.norm(OAc(Ac, Fl, Fr, O, lam) - ncon((Al, OC(C, Fl, Fr)), ([-1, -2, 1], [1, -3])))
        Al, Ar, Ac, C = AlPrime, ArPrime, AcPrime, cPrime
        print(delta)
        i += 1
        if delta < tol:
            flag = 0
    print('Time for VUMPS optimization:', time()-t0, 's')
    print('Iterations needed:', i)
    freeEnergy = freeEnergyDensity(beta, lam)
    freeEnergyExact = isingExact(beta, 1)[1]
    print('Computed free energy:', freeEnergy)
    print('Exact free energy:', freeEnergyExact)
