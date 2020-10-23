from tutorialFunctions import *
import matplotlib.pyplot as plt
from time import time

plotMagnetization = False
plotError = True
Tc = 2.2691853142130194
# to calculate magnetization curve
if plotMagnetization:
    D = 12
    d = 2
    J=1
    
    print('Bond dimension: D =', D)
    A = createMPS(D, d)
    Al, Ar, Ac, C = mixedCanonical(A)
    # optimization parameters
    tol = 1e-5
    
    T_array = np.linspace(0.2,3,20)
    magnetizations = []
    magnetizations_exact = []
    frees = []
    frees_exact = []
    for T in T_array:
        beta = 1/T
        O = isingO(beta, J)
        t0 = time()
        delta = 1e-4
        flag = 1
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
                
        print('##############')
        print('T={}'.format(T))
        print('Time for VUMPS optimization:', time()-t0, 's')
        print('Iterations needed:', i)
        freeEnergy = freeEnergyDensity(beta, lam)
        freeEnergyExact = isingExact(beta, J)[1]
        print('Computed free energy:', freeEnergy)
        print('Exact free energy:', freeEnergyExact)
        print('##############')
    
        magnetizations.append(isingMagnetization(beta, J, Ac, Fl, Fr)/isingZ(beta, J, Ac, Fl, Fr))
        magnetizations_exact.append(isingExact(beta, J)[0])
        
        frees.append(freeEnergyDensity(beta, lam))
        frees_exact.append(isingExact(beta, J)[1])
    
    plt.figure()
    plt.xlabel(r'$T$')
    plt.ylabel(r'$<M>$')
    plt.plot([T for T in T_array], magnetizations, label = 'D={}'.format(D))
    plt.plot([T for T in T_array], magnetizations_exact, label = 'exact')
    plt.figure()
    plt.xlabel(r'$T$')
    plt.ylabel(r'$f$')
    plt.plot([T for T in T_array], frees, label = 'D={}'.format(D))
    plt.plot([T for T in T_array], frees_exact, label = 'exact')
    plt.legend()
    
# to calculate the error as function of bond dimension
if plotError:
    bondDimensions = np.arange(2,10)
    T = 0.9999999*Tc
    beta = 1/T # critical point: 0.440686793509772
    errors, times = [], []
    tol = 1e-5
    for D in bondDimensions:
        d = 2
        D = int(D)
        print('Bond dimension: D =', D)
        
        A = createMPS(D, d)
        Al, Ar, Ac, C = mixedCanonical(A)
        O = isingO(beta, 1)
        delta = 1e-4
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
        errors.append(np.abs((freeEnergy-freeEnergyExact)/freeEnergyExact))
        times.append(time()-t0)
    plt.figure()
    plt.xlabel(r'bond dimension $D$')
    plt.ylabel(r'relative error on $f$')
    plt.title(r'$T=${:.7f}$Tc$'.format(T/Tc))
    plt.plot([int(D) for D in bondDimensions],errors)
    plt.figure()
    plt.xlabel(r'bond dimension $D$')
    plt.ylabel(r'computation time t (s)')
    plt.title(r'$T=${:.7f}$Tc$'.format(T/Tc))
    plt.plot([int(D) for D in bondDimensions],times)
