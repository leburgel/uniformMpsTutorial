from tutorialFunctions import *
import matplotlib.pyplot as plt
from time import time
'''
def Z(beta, J, Ac, Fl, Fr):
    TN = ncon((Fl, Ac, O_tensor(beta, J), np.conj(Ac), Fr), ([1, 3, 2], [2,7,5],[3,7,8,6],[1,6,4], [5,8,4]))
    return TN
'''
D = 12
d = 2
A = createMPS(D,d)
Al, Ar, Ac, C = mixedCanonical(A)
J=1

# optimization parameters
delta = 1e-4
tol = 1e-3
flag = 1

T_array = np.linspace(0.2,3,20)
magnetizations = []
magnetizations_exact = []
for T in T_array:
    beta = 1/T
    O = isingO(beta, J)
    while flag:
        lam, Fl = leftFixedPointMPO(Al, O, delta)
        _ , Fr = rightFixedPointMPO(Ar, O, delta)
        Fl /= overlapFixedPointsMPO(Fl, Fr, C)
        lam = np.real(lam)[0]
        AcPrime, cPrime = calcNewCenterMPO(Ac, C, Fl, Fr, O, lam, delta)
        AlPrime, ArPrime, AcPrime, cPrime = minAcC(AcPrime, cPrime)
        delta = np.linalg.norm(OAc(Ac, Fl, Fr, O, lam) - ncon((Al, OC(C, Fl, Fr)), ([-1, -2, 1], [1, -3])))
        Al, Ar, Ac, C = AlPrime, ArPrime, AcPrime, cPrime
        i += 1
        if delta < tol:
            flag = 0
    print('##############')
    print('T={}'.format(T))
    print('Time for VUMPS optimization:', time()-t0, 's')
    print('Iterations needed:', i)
    freeEnergy = freeEnergyDensity(beta, lam)
    freeEnergyExact = isingExact(beta, 1)[2]
    print('Computed free energy:', freeEnergy)
    print('Exact free energy:', freeEnergyExact)
    print('##############')

    magnetizations.append(isingMagnetization(beta, J, Ac, Fl, Fr)/isingZ(beta, J, Ac, Fl, Fr))
    magnetizations_exact.append(isingExact(beta, J)[0])
    
plt.xlabel(r'$T$')
plt.ylabel(r'$<M>$')
plt.plot([T for T in T_array], magnetizations, label = 'D={}'.format(D))
plt.plot([T for T in T_array], magnetizations_exact, label = 'exact')
plt.legend()