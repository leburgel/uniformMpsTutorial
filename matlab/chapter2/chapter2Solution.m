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


%% Variational optimization of spin-1 Heisenberg Hamiltonian with gradient descent in uniform gauge

% coupling strengths
Jx = -1; Jy = -1; Jz = -1; hz = 0; % Heisenberg antiferromagnet
% Heisenberg Hamiltonian
h = Heisenberg(Jx, Jy, Jz, hz);

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

%% spin-1 Heisenberg Hamiltonian

function h = HeisenbergHamiltonian(Jx, Jy, Jz, hz)
    % spin-1 angular momentum operators
    Sx = [0 1 0; 1 0 1; 0 1 0] / sqrt(2);
    Sy = [0 -1 0; 1 0 -1; 0 1 0] * 1i / sqrt(2);
    Sz = [1 0 0; 0 0 0; 0 0 -1]; 
    % Heisenberg Hamiltonian
    h = -Jx*ncon({Sx, Sx}, {[-1 -3], [-2 -4]}) - Jy*ncon({Sy, Sy}, {[-1 -3], [-2 -4]}) - Jz*ncon({Sz, Sz}, {[-1 -3], [-2 -4]})...
            - hz*ncon({Sz, eye(3)}, {[-1 -3], [-2 -4]}) - hz*ncon({eye(3), eye(3)}, {[-1 -3], [-2 -4]});
end

%% MPS optimization in uniform gauge using gradient descent

function g = EnergyGradient(A, l, r, htilde)
    D = size(A, 1);
    % center terms first; easy ones
    centerTerm1 = ncon({l, r, A, A, conj(A), htilde}, {...
        [6 1],...
        [5 -3],...
        [1 3 2],...
        [2 4 5],...
        [6 7 -1],...
        [3 4 7 -2]});
    centerTerm2 = ncon({l, r, A, A, conj(A), htilde}, {...
        [-1 1],...
        [5 7],...
        [1 3 2],...
        [2 4 5],...
        [-3 6 7],...
        [3 4 -2 6]});
    
    % left environment
    xL =  ncon({l, A, A, conj(A), conj(A), htilde}, {...
        [5 1],...
        [1 3 2],...
        [2 4 -2],...
        [5 6 7],...
        [7 8 -1],...
        [3 4 6 8]});
    handleL = @(v) reshape(reshape(v, [D D]) - ncon({reshape(v, [D D]), A, conj(A)}, {[3 1], [1 2 -2], [3 2 -1]}) + trace(reshape(v, [D D])*r) * l, [], 1);
    Lh = reshape(gmres(handleL, reshape(xL, [], 1)), [D D]);
    leftEnvTerm = ncon({Lh, A, r}, {[-1 1], [1 -2 2], [2 -3]});
    
    % right environment
    xR =  ncon({r, A, A, conj(A), conj(A), htilde}, {...
        [4 5],...
        [-1 2 1],...
        [1 3 4],...
        [-2 8 7],...
        [7 6 5],...
        [2 3 8 6]});
    handleR = @(v) reshape(reshape(v, [D D]) - ncon({reshape(v, [D D]), A, conj(A)}, {[1 3], [-1 2 1], [-2 2 3]}) + trace(l*reshape(v, [D D])) * r, [], 1);
    Rh = reshape(gmres(handleR, reshape(xR, [], 1)), [D D]);
    rightEnvTerm = ncon({Rh, A, l}, {[1 -3], [2 -2 1], [-1 2]});
    
    % construct gradient out of these 4 terms
    g = 2 * (centerTerm1 + centerTerm2 + leftEnvTerm + rightEnvTerm);
end

function [e, g] = EnergyDensity(A, h)
    d = size(A, 2);
    [A, l, r] = NormalizeMPS(A); % normalize MPS
    e = real(ExpvTwoSiteUniform(A, l, r, h)); % compute energy density of MPS (discard numerical imaginary artefact)
    htilde = h - e * ncon({eye(d), eye(d)}, {[-1 -3], [-2 -4]}); % regularized energy density
    g = EnergyGradient(A, l, r, htilde); % calculate gradient of energy density
end

function [e, g] = EnergyWrapper(varA, h, D, d)
    % just wrapper around the EnergyDensity function that takes MPS tensor as a vector and returns the gradient as a vector
    A = complex(reshape(varA(1:D^2*d), [D d D]), reshape(varA(D^2*d+1:end), [D d D]));
    [e, g] = EnergyDensity(A, h);
    g = [reshape(real(g), [], 1); reshape(imag(g), [], 1)];
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

function vprime = H_AC(v, AL, AR, Rh, Lh, htilde)
    % map in equation (131) in the lecture notes, acting on some three-legged tensor 'v'
    centerTerm1 = ncon({AL, v, conj(AL), htilde}, {...
        [4 2 1],...
        [1 3 -3],...
        [4 5 -1],...
        [2 3 5 -2]});
    
    centerTerm2 = ncon({v, AR, conj(AR), htilde}, {...
        [-1 2 1],...
        [1 3 4],...
        [-3 5 4],...
        [2 3 -2 5]});
    
    leftEnvTerm = ncon({Lh, v}, {[-1 1], [1 -2 -3]});
    
    rightEnvTerm = ncon({v, Rh}, {[-1 -2 1], [1 -3]});
    
    vprime = centerTerm1 + centerTerm2 + leftEnvTerm + rightEnvTerm;
end

function vprime = H_C(v, AL, AR, Rh, Lh, htilde)
    % map in equation (132) in the lecture notes, acting on some two-legged tensor 'v'
    centerTerm = ncon({AL, v, AR, conj(AL), conj(AR), htilde}, {...
        [5 3 1],...
        [1 2],...
        [2 4 7],...
        [5 6 -1],...
        [-2 8 7],...
        [3 4 6 8]});
    
    leftEnvTerm = Lh * v;
    
    rightEnvTerm = v * Rh;
    
    vprime = centerTerm + leftEnvTerm + rightEnvTerm;
end

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
