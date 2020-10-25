% Matlab script for chapter 1 of Bad Honnef tutorial on "Tangent space
% methods for Tangent-space methods for uniform matrix product states",
% based on the lecture notes: https://arxiv.org/abs/1810.07006
% 
% Detailed explanations of all the different steps can be found in the
% python notebooks for the different chapters. These files provide a canvas
% for a MATLAB implementation that mirrors the contents of the python
% notebooks

%% 2. Finding ground states of local Hamiltonians

% Unlike the notebooks, where function definitions and corresponding checks
% are constructed in sequence, here all checks are placed at the start of
% the script, while all function definitions must be given at the bottom of
% the script


%% 2.2 Gradient descent algorithms


% Variational optimization of spin-1 Heisenberg Hamiltonian through
% minimization of the gradient in uniform gauge

% coupling strengths
Jx = -1; Jy = -1; Jz = -1; hz = 0; % Heisenberg antiferromagnet
% Heisenberg Hamiltonian
h = Heisenberg(Jx, Jy, Jz, hz);

% initialize bond dimension, physical dimension
D = 6;
d = 3;

% initialize random MPS
A = createMPS(D, d);
A = normaliseMPS(A);


% Minimization of the energy through naive gradient descent
tol = 1e-3;
disp('Gradient descent optimization:\n')
tic;
[E1, A1] = groundStateGradDescent(h, D, 0.1, A, tol, 1e4);
t1 = toc;
fprintf('Time until convergence: %ds', t1)
fprintf('Computed energy: %d\n', E1)

% Minimization of the energy using naive gradient descent:
tol = 1e-5;
disp('Optimization using fminunc:\n')
tic
[E2, A2] = groundStateMinimise(h, D, A, tol);
t2 = toc;
fprintf('Time until convergence: %ds', t2, 's')
fprintf('Computed energy: %d\n', E2)


% % most naive approach: converges to same energy as fminunc
% A = randcomplex(D, d, D);
% tl = 1e-4;
% epsilon = 0.1;
% flag = true;
% while flag
%     [e, g] = EnergyDensity(A, h);
%     e
%     Aprime = A - epsilon * g;
%     if ArrayIsEqual(A, Aprime, tl)
%         flag = false;
%     else
%         A = Aprime;
%     end
% end

% running now, converges to the same energy every time
ReA = rand(D, d, D);
ImA = rand(D, d, D);
varA = [reshape(ReA, [], 1); reshape(ImA, [], 1)];
EnergyHandle = @(varA) EnergyWrapper(varA, h, D, d);
options = optimoptions('fminunc', 'SpecifyObjectiveGradient', true);
tic
[Aopt, e] = fminunc(EnergyHandle, varA, options);
toc
Aopt = complex(reshape(Aopt(1:D^2*d), [D d D]), reshape(Aopt(D^2*d+1:end), [D d D]));
[Aopt, l, r] = NormalizeMPS(Aopt);
ArrayIsEqual(e, ExpvTwoSiteUniform(Aopt, l, r, h), tol) % just to be extra sure...

% gradient descent and fminunc seem to be working now, but sometimes convergence criteria are too strict for fminunc so it just quits at some point


%% Variational optimization of spin-1 Heisenberg Hamiltonian with VUMPS
D = 12;
d = 3;
% tolerance for VUMPS algorithm
tol = 1e-3;
% coupling strengths
Jx = -1; Jy = -1; Jz = -1; hz = 0;
% Heisenberg Hamiltonian
h = HeisenbergHamiltonian(Jx, Jy, Jz, hz); % Heisenberg antiferromagnet

A = randcomplex(D, d, D); % initialize random MPS tensor
[AL, AR, AC, C] = MixedCanonical(A); % go to mixed gauge
flag = true;
delta = 1e-4;
tic
i = 0;
while flag
    e = real(ExpvTwoSiteMixed(AC, AL, h)); % current energy density
    fprintf('Current energy: %d\n', e)
    htilde = h - e * ncon({eye(d), eye(d)}, {[-1 -3], [-2 -4]}); % regularized energy density
    Rh = RightEnvMixed(AR, C, htilde, delta);
    Lh = LeftEnvMixed(AL, C, htilde, delta);
    [ACprime, Cprime] = CalculateNewCenter(AL, AR, AC, C, Rh, Lh, htilde, delta);
    [ALprime, ARprime, ACprime, Cprime] = MinAcC(ACprime, Cprime);
    delta = ArrayNorm(H_AC(AC, AL, AR, Rh, Lh, htilde) - ncon({AL, H_C(C, AL, AR, Rh, Lh, htilde)}, {[-1 -2 1], [1 -3]})); % calculate error using new or old AL, AR, Rh, Lh? now using old...
    fprintf('Current error: %d\n', delta)
    AL = ALprime; AR = ARprime; AC = ACprime; C = Cprime; % update
    i = i+1;
    if delta < tol
        flag = false;
    end
end
toc
fprintf('Iterations needed: %i\n', i)

[U, C, V] = svd(C);

svals = diag(C);

plot(svals, 'd');

% converging and finding correct energy for antiferromagnet
% convergence seems a bit slow, why?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECKS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 1.1 Normalisation

d = 3;
D = 5;

% initializing a random MPS:
A = createMPS(D, d);

assert(isequal(size(A), [D, d, D]), 'Generated MPS tensor has incorrect shape.')
assert(~isreal(a), 'MPS tensor should have complex values')


% normalising an MPS through naive diagonalization of the transfer matrix:
A = normaliseMPSNaive(A);
[l, r] = fixedPointsNaive(A);

assert(ArrayIsEqual(l, l', 1e-12), 'left fixed point should be hermitian!')
assert(ArrayIsEqual(r, r', 1e-12), 'left fixed point should be hermitian!')

assert(ArrayIsEqual(l, ncon({A, l, conj(A)}, {[1, 2, -2], [3, 1], [3, 2, -1]}), 1e-12), 'l should be a left fixed point!')
assert(ArrayIsEqual(r, ncon({A, r, conj(A)}, {[-1, 2, 1], [1, 3], [-2, 2, 3]}), 1e-12), 'r should be a right fixed point!')
assert(abs(trace(l*r) - 1) < 1e-12, 'Left and right fixed points should be trace normalised!')


%% 1.2 Gauge fixing

% left and right orthonormalisation through taking square root of fixed points

[L, Al] = leftOrthonormaliseNaive(A, l);
[R, Ar] = rightOrthonormaliseNaive(A, r);

assert(ArrayIsEqual(R * R', r, 1e-12), 'Right gauge does not square to r')
assert(ArrayIsEqual(L' * L, l, 1e-12), 'Left gauge does not sqaure to l')
assert(ArrayIsEqual(eye(D), ncon({Ar, conj(Ar)}, {[-1 1 2], [-2 1 2]}), 1e-9), 'Ar not in right-orthonormal form')
assert(ArrayIsEqual(eye(D), ncon({Al, conj(Al)}, {[1 2 -2], [1 2 -1]}), 1e-9), 'Al not in left-orthonormal form')


% going to mixed gauge
[Al, Ac, Ar, C] = mixedCanonicalNaive(A);

assert(ArrayIsEqual(eye(D), ncon({Ar, conj(Ar)}, {[-1 1 2], [-2 1 2]}), 1e-9), 'Ar not in right-orthonormal form')
assert(ArrayIsEqual(eye(D), ncon({Al, conj(Al)}, {[1 2 -2], [1 2 -1]}), 1e-9), 'Al not in left-orthonormal form')
LHS = ncon({Al, C}, {[-1, -2, 1], [1, -3]});
RHS = ncon({C, Ar}, {[-1, 1], [1, -2, -3]});
assert(ArrayIsEqual(LHS, RHS, 1e-12) && ArrayIsEqual(RHS, Ac, 1e-12), 'Something went wrong in gauging the MPS')


%% 1.3 Truncation of a uniform MPS

[AlTilde, AcTilde, ArTilde, CTilde] = truncateMPS(A, 3);
assert(isequal(size(AlTilde), [3, 3, 3]), 'Something went wrong in truncating the MPS')


%% 1.4 Algorithms for finding canonical forms

A = createMPS(D, d);

% normalising an MPS through action of transfer matrix on a left and right matrix as a function handle:
A = normaliseMPS(A);
[l, r] = fixedPoints(A);

assert(ArrayIsEqual(l, l', 1e-12), 'left fixed point should be hermitian!')
assert(ArrayIsEqual(r, r', 1e-12), 'left fixed point should be hermitian!')

assert(ArrayIsEqual(l, ncon({A, l, conj(A)}, {[1, 2, -2], [3, 1], [3, 2, -1]}), 1e-12), 'l should be a left fixed point!')
assert(ArrayIsEqual(r, ncon({A, r, conj(A)}, {[-1, 2, 1], [1, 3], [-2, 2, 3]}), 1e-12), 'r should be a right fixed point!')
assert(abs(trace(l*r) - 1) < 1e-12, 'Left and right fixed points should be trace normalised!')


% gauging an MPS through iterative QR decompositions:
[Al, Ac, Ar, C] = mixedCanonical(A);
% [R, Ar] = rightOrthonormalise(A);
% [L, Al] = leftOrthonormalise(A);

assert(ArrayIsEqual(eye(D), ncon({Ar, conj(Ar)}, {[-1 1 2], [-2 1 2]}), 1e-12), 'Ar not in right-orthonormal form')
assert(ArrayIsEqual(eye(D), ncon({Al, conj(Al)}, {[1 2 -2], [1 2 -1]}), 1e-12), 'Al not in left-orthonormal form')
LHS = ncon({Al, C}, {[-1, -2, 1], [1, -3]});
RHS = ncon({C, Ar}, {[-1, 1], [1, -2, -3]});
assert(ArrayIsEqual(LHS, RHS, 1e-12) && ArrayIsEqual(RHS, Ac, 1e-12), 'Something went wrong in gauging the MPS')


%% 1.5 Computing expectation values
A = createMPS(D, d);
A = normaliseMPS(A);
[l, r] = fixedPoints(A);
[Al, Ac, Ar, C] = mixedCanonical(A);

O1 = rand(d,d) + 1i * rand(d,d);
expVal = expVal1Uniform(O1, A, l, r);
expValMix = expVal1Mixed(O1, Ac);
diff = abs(expVal - expValMix);
assert(diff < 1e-12, 'different gauges give different values?')

O2 =rand(d, d, d, d) + 1i *rand(d, d, d, d);
expVal = expVal2Uniform(O2, A, l, r);
expValGauge = expVal2Mixed(O2, Ac, Ar);
expValGauge2 = expVal2Mixed(O2, Al, Ac);

diff1 = abs(expVal - expValGauge);
diff2 = abs(expVal - expValGauge2);
assert(diff1 < 1e-12 && diff2 < 1e-12, 'different gauges give different values?')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION DEFINITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 2.1 The gradient


function hTilde = reducedHamUniform(h, A, l, r)
    % Regularise Hamiltonian such that its expectation value is 0.
    % 
    %     Parameters
    %     ----------
    %     h : array (d, d, d, d)
    %         Hamiltonian that needs to be reduced,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    %     A : array (D, d, D)
    %         normalised MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    % 
    %     Returns
    %     -------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    
    d = size(A, 2);
    % calculate fixed points if not supplied
    if nargin < 4
        [l, r] = fixedPoints(A);
    end
    % calculate expectation value of energy
    e = real(expVal2Uniform(h, A, l, r));
    % substract from hamiltonian
    hTilde = h - e * ncon({eye(d), eye(d)}, {[-1, -3], [-2, -4]});
end


function [term1, term2] = gradCenterTerms(hTilde, A, l, r)
    % Calculate the value of the center terms.
    % 
    %     Parameters
    %     ----------
    %     hTilde : np.array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    %     A : np.array (D, d, D)
    %         normalised MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    % 
    %     Returns
    %     -------
    %     term1 : np.array(D, d, D)
    %         first term of gradient,
    %         ordered left-mid-right.
    %     term2 : np.array(D, d, D)
    %         second term of gradient,
    %         ordered left-mid-right.
    
    % calculate fixed points if not supplied
    if nargin < 4
        [l, r] = fixedPoints(A);
    end
    % calculate first contraction
    term1 = ncon({l, r, A, A, conj(A), hTilde}, {[-1, 1], [5, 7], [1, 3, 2], [2, 4, 5], [-3, 6, 7], [3, 4, -2, 6]});
    % calculate second contraction
    term2 = ncon({l, r, A, A, conj(A), hTilde}, {[6, 1], [5, -3], [1, 3, 2], [2, 4, 5], [6, 7, -1], [3, 4, 7, -2]});
end


function vNew = EtildeRight(A, l, r, v)
    % Implement the action of (1 - Etilde) on a right vector v.
    % 
    %     Parameters
    %     ----------
    %     A : np.array (D, d, D)
    %         normalised MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    %     v : np.array(D**2)
    %         right matrix of size (D, D) on which
    %         (1 - Etilde) acts,
    %         given as a vector of size (D**2,)
    % 
    %     Returns
    %     -------
    %     vNew : np.array(D**2)
    %         result of action of (1 - Etilde)
    %         on a right matrix,
    %         given as a vector of size (D**2,)

    D = size(A, 1);
    % reshape to matrix
    v = reshape(v, [D D]);
    % transfermatrix contribution
    transfer = ncon({A, conj(A), v}, {[-1, 2, 1], [-2, 2, 3], [1, 3]});
    % fixed point contribution
    fixed = trace(l * v) * r;
    % sum these with the contribution of the identity, reshape result to vector
    vNew = reshape(v - transfer + fixed, [], 1);
end


function Rh = RhUniform(hTilde, A, l, r)
    % Find the partial contraction for Rh.
    % 
    %     Parameters
    %     ----------
    %     hTilde : np.array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     A : np.array (D, d, D)
    %         normalised MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    % 
    %     Returns
    %     -------
    %     Rh : np.array(D, D)
    %         result of contraction,
    %         ordered top-bottom.
    
    D = size(A, 1);
    % calculate fixed points if not supplied
    if nargin < 4
        [l, r] = fixedPoints(A);
    end
    % construct b, which is the matrix to the right of (1 - E)^P in the figure in the notebook
    b = ncon({r, A, A, conj(A), conj(A), hTilde}, {[4, 5], [-1, 2, 1], [1, 3, 4], [-2, 8, 7], [7, 6, 5], [2, 3, 8, 6]});
    % solve Ax = b for x, where x is Rh, and reshape result to matrix
    A = @(v) EtildeRight(A, l, r, v);
    Rh = reshape(gmres(A, reshape(b, [], 1)), [D D]);
end


function leftTerms = gradLeftTerms(hTilde, A, l, r)
    % Calculate the value of the left terms.
    % 
    %     Parameters
    %     ----------
    %     hTilde : np.array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     A : np.array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    % 
    %     Returns
    %     -------
    %     leftTerms : np.array(D, d, D)
    %         left terms of gradient,
    %         ordered left-mid-right.
    
    % calculate fixed points if not supplied
    if nargin < 4
        [l, r] = fixedPoints(A);
    end    
    % calculate partial contraction
    Rh = RhUniform(hTilde, A, l, r);
    % calculate full contraction
    leftTerms = ncon({Rh, A, l}, {[1, -3], [2, -2, 1], [-1, 2]});
end


function vNew = EtildeLeft(A, l, r, v)
    % Implement the action of (1 - Etilde) on a left vector matrix v.
    % 
    %     Parameters
    %     ----------
    %     A : np.array (D, d, D)
    %         normalised MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    %     v : np.array(D**2)
    %         right matrix of size (D, D) on which
    %         (1 - Etilde) acts,
    %         given as a vector of size (D**2,)
    % 
    %     Returns
    %     -------
    %     vNew : np.array(D**2)
    %         result of action of (1 - Etilde)
    %         on a left matrix,
    %         given as a vector of size (D**2,)
    
    D = size(A, 1);
    % reshape to matrix
    v = reshape(v, [D D]);
    % transfer matrix contribution
    transfer = ncon({v, A, conj(A)}, {[3, 1], [1, 2, -2], [3, 2, -1]});
    % fixed point contribution
    fixed = trace(v * r) * l;
    % sum these with the contribution of the identity, reshape result to vector
    vNew = reshape(v - transfer + fixed, [], 1);
end


function Lh = LhUniform(hTilde, A, l, r)
    % Find the partial contraction for Lh.
    % 
    %     Parameters
    %     ----------
    %     hTilde : np.array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     A : np.array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    % 
    %     Returns
    %     -------
    %     Lh : np.array(D, D)
    %         result of contraction,
    %         ordered bottom-top.

    D = size(A, 1);
    % calculate fixed points if not supplied
    if nargin < 4
        [l, r] = fixedPoints(A);
    end
    % construct b, which is the matrix to the right of (1 - E)^P in the figure in the notebook
    b = ncon({l, A, A, conj(A), conj(A), hTilde}, {[5, 1], [1, 3, 2], [2, 4, -2], [5, 6, 7], [7, 8, -1], [3, 4, 6, 8]});
    % solve Ax = b for x, where x is Lh, and reshape result to matrix
    A = @(v) EtildeLeft(A, l, r, v);
    Lh = reshape(gmres(A, reshape(b, [], 1)), [D D]);
end


function rightTerms =  gradRightTerms(hTilde, A, l, r)
    % Calculate the value of the right terms.
    % 
    %     Parameters
    %     ----------
    %     hTilde : np.array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     A : np.array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    % 
    %     Returns
    %     -------
    %     rightTerms : np.array(D, d, D)
    %         right terms of gradient,
    %         ordered left-mid-right.
    
    % calculate fixed points if not supplied
    if nargin < 4
        [l, r] = fixedPoints(A);
    end    
    % calculate partial contraction
    Lh = LhUniform(hTilde, A, l, r);
    % calculate full contraction
    rightTerms = ncon({Lh, A, r}, {[-1, 1], [1, -2, 2], [2, -3]});
end


function grad = gradient(h, A, l, r)
    % Calculate the gradient of the expectation value of h @ MPS A.
    % 
    %     Parameters
    %     ----------
    %     h : np.array (d, d, d, d)
    %         Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     A : np.array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : np.array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalised.
    %     r : np.array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalised.
    % 
    %     Returns
    %     -------
    %     grad : np.array(D, d, D)
    %         Gradient,
    %         ordered left-mid-right.
    
    % calculate fixed points if not supplied
    if nargin < 4
        [l, r] = fixedPoints(A);
    end    
    % regularise Hamiltonian
    hTilde = reducedHamUniform(h, A, l, r);
    % find terms
    [centerTerm1, centerTerm2] = gradCenterTerms(hTilde, A, l, r);
    leftTerms = gradLeftTerms(hTilde, A, l, r);
    rightTerms = gradRightTerms(hTilde, A, l, r);
    grad = 2 * (centerTerm1 + centerTerm2 + leftTerms + rightTerms);
end


function [E, A] = groundStateGradDescent(h, D, eps, A0, tol, maxIter)
    % Find the ground state using gradient descent.
    % 
    %     Parameters
    %     ----------
    %     h : np.array (d, d, d, d)
    %         Hamiltonian to minimise,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    %     D : int
    %         Bond dimension
    %     eps : float
    %         Stepsize.
    %     A0 : np.array (D, d, D)
    %         normalised MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         initial guess.
    %     tol : float
    %         Tolerance for convergence criterium.
    %     maxIter : int
    %         Maximum number of iterations.
    % 
    %     Returns
    %     -------
    %     E : float
    %         expectation value @ minimum
    %     A : np.array(D, d, D)
    %         ground state MPS,
    %         ordered left-mid-right.

    d = size(h, 1);
    if nargin < 6
        maxIter = 1e5;
    end
    if nargin < 5
        tol = 1e-3;
    end
    if nargin < 4
        A0 = createMPS(D, d);
        A0 = normaliseMPS(A0);
    end
    % calculate gradient
    g = gradient(h, A0);
    A = A0;
    i = 0;
    while ~(all(all(all(abs(g) < tol))))
        % do a step
        A = A - eps * g;
        A = normaliseMPS(A);
        i = i + 1;
        
        if ~(mod(i,100))
            E = real(expVal2Uniform(h, A));
            fprintf('Current energy: %d', E)
        end
        % calculate new gradient
        g = gradient(h, A);
        if i > maxIter
            disp('Warning: gradient descent did not converge!')
        end
    end
    % calculate ground state energy
    E = real(expVal2Uniform(h, A));
end


function A = unwrapper(varA, D, d)
    % Unwraps real MPS vector to complex MPS tensor.
    % 
    %     Parameters
    %     ----------
    %     varA : array(2 * D * d * D)
    %         MPS tensor in real vector form.
    %     D : int
    %         Bond dimension.
    %     d : int
    %         Physical dimension.
    % 
    %     Returns
    %     -------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
        
    % unpack real and imaginary part
    Areal = varA(1:D^2 * d);
    Aimag = varA(D^2 * d:end);
    A = reshape(complex(Areal, Aimag), [D d D]);
end


function varA = wrapper(A)
    % Wraps MPS tensor to real MPS vector.
    % 
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor,
    %         ordered left-bottom-right
    % 
    %     Returns
    %     -------
    %     varA : array(2 * D * d * D)
    %         MPS tensor in real vector form.
        
    % split into real and imaginary part
    Areal = reshape(real(A), [], 1);
    Aimag = reshape(imag(A), [], 1);
    % combine into vector
    varA = [Areal; Aimag];
end
        

function [e, g] = energyDensity(varA, D, d)
    %     Function to optimize via fminunc.
    % 
    %         Parameters
    %         ----------
    %         varA : np.array(2 * D * d * D)
    %             MPS tensor in real vector form.
    % 
    %         Returns
    %         -------
    %         e : float
    %             function value @varA
    %         g : np.array(2 * D * d * D)
    %             gradient vector @varA
        
    % unwrap varA
    A = unwrapper(varA, D, d);
    A = normaliseMPS(A);
    % calculate fixed points
    [l, r] = fixedPoints(A);
    % calculate function value and gradient
    e = real(expVal2Uniform(h, A, l, r));
    g = gradient(h, A, l, r);
    % wrap g
    g = wrapper(g);
end


function [E, Aopt] = groundStateMinimise(h, D, A0, tol)
    % Find the ground state using the MATLAB fminunc minimizer.
    % 
    %     Parameters
    %     ----------
    %     h : array (d, d, d, d)
    %         Hamiltonian to minimise,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    %     D : int
    %         Bond dimension
    %     A0 : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         initial guess.
    %     tol : float
    %         Relative convergence criterium.
    % 
    %     Returns
    %     -------
    %     E : float
    %         expectation value @ minimum
    %     A : array(D, d, D)
    %         ground state MPS,
    %         ordered left-mid-right.
    
    d = size(h, 1);
    if nargin < 5
        tol = 1e-4;
    end
    if nargin < 4
        A0 = createMPS(D, d);
        A0 = normaliseMPS(A0);
    end
    varA0 = wrapper(A0);
    % calculate minimum
    energyHandle = @(varA) energyDensity(varA, D, d);
    options = optimoptions('fminunc', 'SpecifyObjectiveGradient', true, 'Display', 'iter', 'OptimalityTolerance', tol);
    [varAopt, E] = fminunc(energyHandle, varA0, options);
    Aopt = normaliseMPS(unwrapper(varAopt, D, d));
end


function h = Heisenberg(Jx, Jy, Jz, hz)
    % Construct the spin-1 Heisenberg Hamiltonian for given couplings.
    % 
    %     Parameters
    %     ----------
    %     Jx : float
    %         Coupling strength in x direction
    %     Jy : float
    %         Coupling strength in y direction
    %     Jy : float
    %         Coupling strength in z direction
    %     hz : float
    %         Coupling for Sz terms
    % 
    %     Returns
    %     -------
    %     h : array (3, 3, 3, 3)
    %         Spin-1 Heisenberg Hamiltonian.
    
    % spin-1 angular momentum operators
    Sx = [0 1 0; 1 0 1; 0 1 0] / sqrt(2);
    Sy = [0 -1 0; 1 0 -1; 0 1 0] * 1i / sqrt(2);
    Sz = [1 0 0; 0 0 0; 0 0 -1]; 
    % Heisenberg Hamiltonian
    h = -Jx*ncon({Sx, Sx}, {[-1 -3], [-2 -4]}) - Jy*ncon({Sy, Sy}, {[-1 -3], [-2 -4]}) - Jz*ncon({Sz, Sz}, {[-1 -3], [-2 -4]})...
            - hz*ncon({Sz, eye(3)}, {[-1 -3], [-2 -4]}) - hz*ncon({eye(3), eye(3)}, {[-1 -3], [-2 -4]});
end


%% 2.3 VUMPS


function hTilde = reducedHamMixed(h, Ac, Ar)
    % Regularise Hamiltonian such that its expectation value is 0.
    % 
    %     Parameters
    %     ----------
    %     h : array (d, d, d, d)
    %         Hamiltonian that needs to be reduced,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    %     Ac : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauged.
    %     Ar : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right gauged.
    % 
    %     Returns
    %     -------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    
    d = size(Ac, 2);
    % calculate expectation value
    e = real(expVal2Mixed(h, Ac, Ar));
    % substract from hamiltonian
    hTilde = h - e * ncon({eye(d), eye(d)}, {[-1, -3], [-2, -4]});
end


function Rh = RhMixed(hTilde, Ar, C, tol)
    % Calculate Rh, for a given MPS in mixed gauge.
    % 
    %     Parameters
    %     ----------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     Ar : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right-orthonormal.
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right.
    %     tol : float, optional
    %         tolerance for gmres
    % 
    %     Returns
    %     -------
    %     Rh : array(D, D)
    %         result of contraction,
    %         ordered top-bottom.
    
    D = size(Ar, 1);
    if nargin < 4
        tol = 1e-4;
    end
    % construct fixed points for Ar
    l = C' * C; % left fixed point of right transfer matrix
    r = eye(D); % right fixed point of right transfer matrix: right orthonormal
    % construct b
    b = ncon({Ar, Ar, np.conj(Ar), np.conj(Ar), hTilde}, {[-1, 2, 1], [1, 3, 4], [-2, 7, 6], [6, 5, 4], [2, 3, 7, 5]});
    % solve Ax = b for x
    A = @(v) EtildeRight(Ar, l, r, v);
    [Rh, ~] = gmres(A, reshape(b, [], 1), [], tol);
    Rh = reshape(Rh, [D D]);
end


function Lh = LhMixed(hTilde, Al, C, tol)
    % Calculate Lh, for a given MPS in mixed gauge.
    % 
    %     Parameters
    %     ----------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     Al : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left-orthonormal.
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right.
    %     tol : float, optional
    %         tolerance for gmres
    % 
    %     Returns
    %     -------
    %     Lh : array(D, D)
    %         result of contraction,
    %         ordered bottom-top.
    
    D = size(Al, 1);
    if nargin < 4
        tol = 1e-4;
    end
    % construct fixed points for Ar
    l = eye(D); % left fixed point of right transfer matrix
    r = C * C'; % right fixed point of right transfer matrix: right orthonormal
    % construct b
    b = ncon({Al, Al, np.conj(Al), np.conj(Al), hTilde}, {[4, 2, 1], [1, 3, -2], [4, 5, 6], [6, 7, -1], [2, 3, 5, 7]});
    % solve Ax = b for x
    A = @(v) EtildeLeft(Al, l, r, v);
    [Lh, ~] = gmres(A, reshape(b, [], 1), [], tol);
    Lh = reshape(Lh, [D D]);
end


function H_AcV = H_Ac(hTilde, Al, Ar, Lh, Rh, v)
    % Action of the effective Hamiltonian for Ac (131) on a vector.
    % 
    %     Parameters
    %     ----------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     Al : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left-orthonormal.
    %     Ar : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right-orthonormal.
    %     Lh : array(D, D)
    %         left environment,
    %         ordered bottom-top.
    %     Rh : array(D, D)
    %         right environment,
    %         ordered top-bottom.
    %     v : np.array(D, d, D)
    %         Tensor of size (D, d, D)
    % 
    %     Returns
    %     -------
    %     H_AcV : array(D, d, D)
    %         Result of the action of H_Ac on the vector v,
    %         representing a tensor of size (D, d, D)

    % first term
    term1 = ncon({Al, v, np.conj(Al), hTilde}, {[4, 2, 1], [1, 3, -3], [4, 5, -1], [2, 3, 5, -2]});
    % second term
    term2 = ncon({v, Ar, np.conj(Ar), hTilde}, {[-1, 2, 1], [1, 3, 4], [-3, 5, 4], [2, 3, -2, 5]});
    % third term
    term3 = ncon({Lh, v}, {[-1, 1], [1, -2, -3]});
    % fourth term
    term4 = ncon({v, Rh}, {[-1, -2, 1], [1, -3]});
    % sum
    H_AcV = term1 + term2 + term3 + term4;
end


function H_CV = H_C(hTilde, Al, Ar, Lh, Rh, v)
    % Action of the effective Hamiltonian for Ac (131) on a vector.
    % 
    %     Parameters
    %     ----------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     Al : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left-orthonormal.
    %     Ar : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right-orthonormal.
    %     Lh : array(D, D)
    %         left environment,
    %         ordered bottom-top.
    %     Rh : array(D, D)
    %         right environment,
    %         ordered top-bottom.
    %     v : array(D, D)
    %         Matrix of size (D, D)
    % 
    %     Returns
    %     -------
    %     H_CV : array(D, D)
    %         Result of the action of H_C on the matrix v.

    % first term
    term1 = ncon({Al, v, Ar, np.conj(Al), np.conj(Ar), hTilde}, {[5, 3, 1], [1, 2], [2, 4, 7], [5, 6, -1], [-2, 8, 7], [3, 4, 6, 8]});
    % second term
    term2 = Lh * v;
    % third term
    term3 = v * Rh;
    % sum
    H_CV = term1 + term2 + term3;
end


function [AcTilde, CTilde] = calcNewCenter(hTilde, Al, Ac, Ar, C, Lh, Rh, tol)
    % Find new guess for Ac and C as fixed points of the maps H_Ac and H_C.
    % 
    %     Parameters
    %     ----------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     Al : array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal.
    %     Ar : array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         right orthonormal.
    %     Ac : array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right,
    %         diagonal.
    %     Lh : array(D, D)
    %         left environment,
    %         ordered bottom-top.
    %     Rh : array(D, D)
    %         right environment,
    %         ordered top-bottom.
    %     tol : float, optional
    %         current tolerance
    % 
    %     Returns
    %     -------
    %     AcTilde : np.array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     CTilde : np.array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right.
    
    D = size(Al, 1);
    d = size(Al, 2);
    if nargin < 8
        tol = 1e-4;
    end
    if nargin < 7
        Rh = RhMixed(hTilde, Ar, C, tol);
    end
    if nargin < 6
        Lh = LhMixed(hTilde, Al, C, tol);
    end
    % calculate new AcTilde
    % wrapper around H_Ac that takes and returns a vector
    handleAc = @(v) reshape(H_Ac(hTilde, Al, Ar, Lh, Rh, v.reshape(D, d, D)), [], 1);
    % compute eigenvector
    [AcTilde, ~] = eigs(handleAc, D^2*d, 1, 'smallestreal', 'Tolerance', tol, 'StartVector', reshape(Ac, [], 1));
    % calculate new CTilde
    % wrapper around H_C that takes and returns a vector
    handleC = @(v) reahape(H_C(hTilde, Al, Ar, Lh, Rh, v.reshape(D, D)), [], 1);
    % compute eigenvector
    [CTilde, ~] = eigs(handleC, D^2, 1, 'smallestreal', 'Tolerance', tol, 'StartVector', reshape(C, [], 1));
    % reshape to tensors of correct size
    AcTilde = reshape(AcTilde, [D d D]);
    CTilde = reshape(CTilde, [D D]);
end


% Polar decomposition is implemented in the function 'poldec' in the folder
% 'AuxiliaryFunctions'


function [Al, Ac, Ar, C] = minAcC(AcTilde, CTilde)
    % Find Al and Ar corresponding to Ac and C, according to algorithm 5 in the lecture notes.
    % 
    %     Parameters
    %     ----------
    %     AcTilde : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         new guess for center gauge. 
    %     CTilde : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right,
    %         new guess for center gauge
    % 
    %     Returns
    %     -------
    %     Al : array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal.
    %     Ar : array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         right orthonormal.
    %     Ac : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauge. 
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right,
    %         center gauge
        
    D = size(AcTilde, 1);
    d = size(AcTilde, 2);
    % polar decomposition of Ac
    [UlAc, ~] = poldec(reshape(AcTilde, [D * d, D]));
    % polar decomposition of C
    [UlC, ~] = poldec(CTilde);
    % construct Al
    Al = reshape(UlAc * UlC', [D d D]);
    % find corresponding Ar, C, and Ac through right orthonormalising Al
    [C, Ar] = rightOrthonormalise(Al);
    nrm = trace(C * C');
    C = C / sqrt(nrm);
    Ac = ncon({Al, C}, {[-1, -2, 1], [1, -3]});
end
    

function delta = gradientNorm(hTilde, Al, Ac, Ar, C, Lh, Rh)
    % Calculate the norm of the gradient.
    % 
    %     Parameters
    %     ----------
    %     hTilde : array (d, d, d, d)
    %         reduced Hamiltonian,
    %         ordered topLeft-topRight-bottomLeft-bottomRight,
    %         renormalised.
    %     Al : array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal.
    %     Ar : array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         right orthonormal.
    %     Ac : np.array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right.
    %     Lh : array(D, D)
    %         left environment,
    %         ordered bottom-top.
    %     Rh : array(D, D)
    %         right environment,
    %         ordered top-bottom.
    % 
    %     Returns
    %     -------
    %     delta : float
    %         norm of the gradient @Al, Ac, Ar, C
    
    % calculate update on Ac and C using maps H_Ac and H_c
    AcUpdate = H_Ac(hTilde, Al, Ar, Lh, Rh, Ac);
    CUpdate = H_C(hTilde, Al, Ar, Lh, Rh, C);
    AlCupdate = ncon({Al, CUpdate}, {[-1, -2, 1], [1, -3]});
    delta = ArrayNorm(AcUpdate - AlCupdate);
end    


function [E, Al, Ac, Ar, C] = vumps(h, D, A0, tol)
    % Find the ground state of a given Hamiltonian using VUMPS.
    % 
    %     Parameters
    %     ----------
    %     h : np.array (d, d, d, d)
    %         Hamiltonian to minimise,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    %     D : int
    %         Bond dimension
    %     A0 : np.array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         initial guess.
    %     tol : float
    %         Relative convergence criterium.
    % 
    %     Returns
    %     -------
    %     E : float
    %         expectation value @ minimum
    %     A : np.array(D, d, D)
    %         ground state MPS,
    %         ordered left-mid-right.
    
    if nargin < 4
        tol = 1e-4;
    end
    if nargin < 3
        A0 = createMPS(D, d);
        A0 = normaliseMPS(A0);
    end
    % go to mixed gauge
    [Al, Ac, Ar, C] = mixedCanonical(A0);
    flag = true;
    delta = 1e-5;
    while flag
        % regularise H
        hTilde = reducedHamMixed(h, Ac, Ar);
        % calculate environments
        Lh = LhMixed(hTilde, Al, C, delta/10);
        Rh = RhMixed(hTilde, Ar, C, delta/10);
        % calculate new center
        [AcTilde, CTilde] = calcNewCenter(hTilde, Al, Ac, Ar, C, Lh, Rh, tol);
        % find Al, Ar from new Ac, C
        [AlTilde, AcTilde, ArTilde, CTilde] = minAcC(AcTilde, CTilde);
        % calculate norm
        delta = gradientNorm(hTilde, Al, Ac, Ar, C, Lh, Rh);
        % check convergence
        if delta < tol
            flag = false;
        end
        % update tensors
        Al = AlTilde; Ac = AcTilde; Ar = ArTilde; C = CTilde;
        % print current energy, optional...
        E = real(expVal2Mixed(h, Ac, Ar));
        fprintf('Current energy: %d', E);
    end
end



%% VUMPS algorithm for 1-dimensional spin chain

function Rh = RightEnvMixed(AR, C, htilde, delta)
    D = size(AR, 1);
    xR =  ncon({AR, AR, conj(AR), conj(AR), htilde}, {...
        [-1 2 1],...
        [1 3 4],...
        [-2 7 6],...
        [6 5 4],...
        [2 3 7 5]});
    % for regularizing right transfer matrix: left fixed point is C'*C
    handleR = @(v) reshape(reshape(v, [D D]) - ncon({reshape(v, [D D]), AR, conj(AR)}, {[1 3], [-1 2 1], [-2 2 3]}) + trace((C'*C)*reshape(v, [D D])) * eye(D), [], 1);
    Rh = reshape(gmres(handleR, reshape(xR, [], 1), [], delta/10), [D D]); % variable tolerance
end

function Lh = LeftEnvMixed(AL, C, htilde, delta)
    D = size(AL, 1);
    xL =  ncon({AL, AL, conj(AL), conj(AL), htilde}, {...
        [4 2 1],...
        [1 3 -2],...
        [4 5 6],...
        [6 7 -1],...
        [2 3 5 7]});
    % for regularizing right left matrix: right fixed point is C*C'
    handleL = @(v) reshape(reshape(v, [D D]) - ncon({reshape(v, [D D]), AL, conj(AL)}, {[3 1], [1 2 -2], [3 2 -1]}) + trace(reshape(v, [D D])*(C*C')) * eye(D), [], 1);
    Lh = reshape(gmres(handleL, reshape(xL, [], 1), [], delta/10), [D D]); % variable tolerance
end

% function vprime = H_AC(v, AL, AR, Rh, Lh, htilde)
%     % map in equation (131) in the lecture notes, acting on some three-legged tensor 'v'
%     centerTerm1 = ncon({AL, v, conj(AL), htilde}, {...
%         [4 2 1],...
%         [1 3 -3],...
%         [4 5 -1],...
%         [2 3 5 -2]});
%     
%     centerTerm2 = ncon({v, AR, conj(AR), htilde}, {...
%         [-1 2 1],...
%         [1 3 4],...
%         [-3 5 4],...
%         [2 3 -2 5]});
%     
%     leftEnvTerm = ncon({Lh, v}, {[-1 1], [1 -2 -3]});
%     
%     rightEnvTerm = ncon({v, Rh}, {[-1 -2 1], [1 -3]});
%     
%     vprime = centerTerm1 + centerTerm2 + leftEnvTerm + rightEnvTerm;
% end

% function vprime = H_C(v, AL, AR, Rh, Lh, htilde)
%     % map in equation (132) in the lecture notes, acting on some two-legged tensor 'v'
%     centerTerm = ncon({AL, v, AR, conj(AL), conj(AR), htilde}, {...
%         [5 3 1],...
%         [1 2],...
%         [2 4 7],...
%         [5 6 -1],...
%         [-2 8 7],...
%         [3 4 6 8]});
%     
%     leftEnvTerm = Lh * v;
%     
%     rightEnvTerm = v * Rh;
%     
%     vprime = centerTerm + leftEnvTerm + rightEnvTerm;
% end

function [ACprime, Cprime] = CalculateNewCenter(AL, AR, AC, C, Lh, Rh, htilde, delta)
    D = size(AL, 1); d = size(AL, 2);
    % compute action of maps (131) and (132) in the notes and pour this into function handle for eigs
    handleAC = @(v) reshape(H_AC(reshape(v, [D d D]), AL, AR, Rh, Lh, htilde), [], 1);
    handleC = @(v) reshape(H_C(reshape(v, [D D]), AL, AR, Rh, Lh, htilde), [], 1);
    % solve eigenvalue problem using 'smallest real' option
    [ACprime, ~] = eigs(handleAC, D^2*d, 1, 'smallestreal', 'Tolerance', delta/10, 'StartVector', reshape(AC, [], 1)); % variable tolerance
    ACprime = reshape(ACprime, [D d D]);
    [Cprime, ~] = eigs(handleC, D^2, 1, 'smallestreal', 'Tolerance', delta/10, 'StartVector', reshape(C, [], 1)); % variable tolerance
    Cprime = reshape(Cprime, [D D]);
end

function [AL, AR, AC, C] = MinAcC(ACprime, Cprime)
    % algorithm 5 from lecture notes, but adapted so that AR and AL are related properly for regularization of left and right transfer matrix
    D = size(ACprime, 1); d = size(ACprime, 2);
    % left polar decomposition
    [UlAC, ~] = qrpos(reshape(ACprime, [D*d, D]));
    [UlC, ~] = qrpos(Cprime);
    AL = reshape(UlAC*UlC', [D d D]);
    % determine AR through right canonization of AL, and extract C and AC from the result
    % this gives consistent set of MPS tensors in mixed gauge
    [AR, C, ~] = RightOrthonormalize(AL);
    % normalize for mixed gauge
    nrm = trace(C*C');
    C = C / sqrt(nrm);
    AC = ncon({AL, C}, {[-1 -2 1], [1 -3]});
end



%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUXILIARY FUNCTIONS, from chapter 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A =  createMPS(D, d)
    A = rand(D, d, D) + 1i * rand(D, d, D);
end


function Anew =  normaliseMPS(A)
    D = size(A, 1);
    handleERight = @(v) reshape(ncon({A, conj(A), reshape(v, [D D])}, {[-1 2 1], [-2 2 3], [1 3]}), [], 1); % construct transfer matrix handle
    lambda = eigs(handleERight, D^2, 1);
    Anew = A / sqrt(lambda);
end


function l = leftFixedPoint(A)
    D = size(A, 1);
    handleELeft = @(v) reshape(ncon({A, conj(A), reshape(v, [D D])}, {[1 2 -2], [3 2 -1], [3 1]}), [], 1); % construct transfer matrix handle
    [l, ~] = eigs(handleELeft, D^2, 1);
    l = reshape(l, [D D]); % fix shape
    % make left fixed point hermitian and positive semidefinite explicitly
    ldag = l';  l = l / sqrt(l(1) / ldag(1));
    l = (l + l') / 2;
    l = l * sign(l(1));
end


function r = rightFixedPoint(A)
    D = size(A, 1);
    handleERight = @(v) reshape(ncon({A, conj(A), reshape(v, [D D])}, {[-1 2 1], [-2 2 3], [1 3]}), [], 1); % construct transfer matrix handle
    [r, ~] = eigs(handleERight, D^2, 1);
    r = reshape(r, [D D]); % fix shape r
    % make right fixed point hermitian and positive semidefinite explicitly
    rdag = r';  r = r / sqrt(r(1) / rdag(1));
    r = (r + r') / 2;
    r = r * sign(r(1));
end


function [l, r] = fixedPoints(A)
    l = leftFixedPoint(A);
    r = rightFixedPoint(A);
    l = l / trace(l*r); % normalise
end


function [R, Ar] = rightOrthonormalise(A, R0, tol, maxIter)
    D = size(A, 1); d = size(A, 2); i = 0;
    if nargin < 4
        maxIter = 1e5;
    end
    if nargin < 3
        tol = 1e-12;
    end
    if nargin < 2
        R = randcomplex(D, D); % initialize random matrix
    else
        R = R0;
    end
    flag = true;
    while flag
        i = i + 1;
        [Ar, Rprime] = lq(reshape(ncon({A, R}, {[-1 -2 1], [1 -3]}), [D, d*D]));
        lambda = ArrayNorm(Rprime);    Rprime = Rprime / lambda;
        if ArrayIsEqual(R, Rprime, tol)
            flag = false;
        else
            R = Rprime;
        end
        if i > maxIter
            disp('Warning, right decomposition has not converged')
            break
        end
    end
    R = Rprime;
    Ar = reshape(Ar, [D d D]);
end


function [L, Al] = leftOrthonormalise(A, L0, tol, maxIter)            
    D = size(A, 1); d = size(A, 2); i = 0;
    if nargin < 4
        maxIter = 1e5;
    end
    if nargin < 3
        tol = 1e-12;
    end
    if nargin < 2
        L = randcomplex(D, D); % initialize random matrix
    else
        L = L0;
    end
    L = L / ArrayNorm(L); % normalize
    flag = true;
    while flag
        i = i + 1;
        [Al, Lprime] = qrpos(reshape(ncon({L, A}, {[-1 1], [1 -2 -3]}), [D*d, D]));
        lambda = ArrayNorm(Lprime);    Lprime = Lprime / lambda;
        if ArrayIsEqual(L, Lprime, tol)
            flag = false;
        else
            L = Lprime;
        end
        if i > maxIter
            disp('Warning, right decomposition has not converged')
            break
        end
    end
    L = Lprime;
    Al = reshape(Al, [D d D]);
end


function [Al, Ac, Ar, C] = mixedCanonical(A, tol)
    if nargin < 2
        tol = 1e-12;
    end
    D = size(A, 1);
    R0 = randcomplex(D, D); L0 = randcomplex(D, D); % initialize random matrices
    [L, Al] = leftOrthonormalise(A, L0, tol);
    [R, Ar] = rightOrthonormalise(A, R0, tol);
    [U, C, V] = svd(L * R);
    % normalize center matrix
    nrm = trace(C * C');
    C = C / sqrt(nrm);
    % compute MPS tensors
    Al = ncon({U', Al, U}, {[-1 1], [1 -2 2], [2 -3]});
    Ar = ncon({V', Ar, V}, {[-1 1], [1 -2 2], [2 -3]});
    Ac = ncon({Al, C}, {[-1 -2 1], [1 -3]});
end


function o = expVal1Uniform(O, A, l, r)
    o = ncon({l, r, A, conj(A), O}, {[4 1], [3 6], [1 2 3], [4 5 6], [2 5]});
end

function o = expVal1Mixed(O, AC)
    o = ncon({AC, conj(AC), O}, {[1 2 3], [1 4 3], [2 4]}, [2 1 3 4]);
end

function o = expVal2Uniform(O, A, l, r)
    o = ncon({l, r, A, A, conj(A), conj(A), O}, {[6 1], [5 10], [1 2 3], [3 4 5], [6 7 8], [8 9 10], [2 4 7 9]});
end

function o = expVal2Mixed(O, Al, Ac)
    o = ncon({Al, Ac, conj(Al), conj(Ac), O}, { [1 2 3], [3 4 5], [1 6 7], [7 8 5], [2 4 6 8]}, [3 2 4 1 6 5 8 7]);
end
