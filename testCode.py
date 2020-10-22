from tutorialFunctions import *

### A first test case for the gradient in python
D = 6
d = 3

H = Heisenberg(-1, -1, -1, 0)

A = createMPS(D, d)
ReA = np.real(A)
ImA = np.imag(A)

if False:
    # calculatee optimal paths for gradient contractions
    r = np.ones((D, D))
    l = np.ones((D, D))
    path = np.einsum_path('ijk,klm,jlqo,rqp,ri,mn->pon', A, A, H, np.conj(A), l, r, optimize='optimal')

    print(path)

# extra haakjes om real(g) en imag(g) in tuple te plaatsen voor concate anders error !!!
varA = np.concatenate((ReA.reshape(-1), ImA.reshape(-1)))

EnergyHandle = partial(energyWrapper, H, D, d)


res = minimize(EnergyHandle, varA, jac=True)
Aopt = res.x
print(res.fun)

import scipy.io
mat = scipy.io.loadmat('Aoptimized.mat')
#print(mat)
Aopt = mat['Aopt']
print(energyDensity(Aopt, H)[0])