import matplotlib.pyplot as plt
import numpy as np
from tutorialFunctions import createMPS, mixedCanonical, O_tensor, partitionLeft,\
overlapPartitionsFlFr, partitionCenter, minAcC, oAc, oC, kronecker
from ncon import ncon

def M_tensor(beta, J):
    S_z = np.array([[1,0],[0,-1]])
    c, s = np.sqrt(np.cosh(beta*J)), np.sqrt(np.sinh(beta*J))
    Q_sqrt = 1/np.sqrt(2) * np.array([[c+s, c-s],[c-s, c+s]])
    delta_new = ncon((S_z, kronecker(2,4)), ([-1,1], [1,-2,-3,-4]))
    M = ncon((Q_sqrt, Q_sqrt, Q_sqrt, Q_sqrt, delta_new), ([-1,1], [-2,2], [-3,3], [-4,4], [1,2,3,4]))
    return M

def Magnetization(beta, J, Ac, Fl, Fr):
    M_exp = ncon((Fl, Ac, M_tensor(beta, J), np.conj(Ac), Fr), ([1, 3, 2], [2,7,5],[3,7,8,6],[1,6,4], [5,8,4]))
    return M_exp

def Z(beta, J, Ac, Fl, Fr):
    TN = ncon((Fl, Ac, O_tensor(beta, J), np.conj(Ac), Fr), ([1, 3, 2], [2,7,5],[3,7,8,6],[1,6,4], [5,8,4]))
    return TN

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
for T in T_array:
    beta = 1/T
    O = O_tensor(beta,J)
    while flag:
        lam, Fl = partitionLeft(Al, O, delta)
        _ , Fr = partitionLeft(Ar, O, delta)
        overlap = overlapPartitionsFlFr(Fl, Fr, C)
        Fl = Fl/overlap
        lam = np.real(lam)[0]
        AcPrime, cPrime = partitionCenter(Ac, C, Fl, Fr, O, lam, delta)
        AlPrime, ArPrime, AcPrime, cPrime = minAcC(AcPrime, cPrime)
        delta = np.linalg.norm(oAc(Ac, Fl, Fr, O, lam) - ncon((Al, oC(C, Fl, Fr)), ([-1, -2, 1], [1, -3])))
        Al = AlPrime
        Ar = ArPrime
        Ac = AcPrime
        C = cPrime
        print(delta)
        if delta < tol:
            flag = 0
            print('T={} succeeded'.format(T))

    magnetizations.append(Magnetization(beta, J, Ac, Fl, Fr)/Z(beta, J, Ac, Fl, Fr))
    
plt.xlabel(r'$T$')
plt.ylabel(r'$<M>$')
plt.plot([T for T in T_array], magnetizations)