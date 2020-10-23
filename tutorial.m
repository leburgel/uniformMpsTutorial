% Matlab script

%% Some intialization

tol = 1e-12;

D = 12;
d = 3;

A = randcomplex(D, d, D); % MPS tensor
E = ncon({A, conj(A)}, {[-1 1 -3], [-2 1 -4]}); % transfer matrix -> implemented in function CreateTransfer

%% Calculate largest magnitude left and right eigenvectors of transfer matrix

% first option: directly using transfer matrix
[r1, lambda1] = eigs(reshape(E, [D^2, D^2]), 1); % right eigenvector
[l1, ltst1] = eigs(reshape(E, [D^2, D^2]).', 1); % left eigenvector
r1 = reshape(r1, [D D]);
l1 = reshape(l1, [D D]).'; % transpose for indexing convention

% second option: smart way using function handle for eigs
% !have to reshape v to and from matrix in function handle!
handleR = @(v) reshape(ncon({A, conj(A), reshape(v, [D D])}, {[-1 2 1], [-2 2 3], [1 3]}), [], 1); % contraction order good, right?
handleL = @(v) reshape(ncon({A, conj(A), reshape(v, [D D])}, {[1 2 -2], [3 2 -1], [3 1]}), [], 1);
[r2, lambda2] = eigs(handleR, D^2, 1); % right eigenvector
[l2, ltst2] = eigs(handleL, D^2, 1); % left eigenvector
r2 = reshape(r2, [D D]);
l2 = reshape(l2, [D D]);

% check if everything worked out: ok
checkEigenv = ArrayIsEqual(r1, r2 * r1(1) / r2(1), tol) && ArrayIsEqual(l1, l2 * l1(1) / l2(1), tol) && ArrayIsEqual(lambda1, lambda2, tol);

% use one of the results for the remainder of the procedure
l = l2; r = r2; lambda = lambda2;


%% Normalization

% normalize MPS tensor
A = A / sqrt(lambda);
E = ncon({A, conj(A)}, {[-1 1 -3], [-2 1 -4]});

% normalize fixed points
tr = ncon({l, r},{[1 2], [2 1]});
l = l / sqrt(tr); r = r / sqrt(tr);

% TODO: explicitly make left and right points hermititian!

% checks to see if this is right: ok
checkr = ArrayIsEqual(r, ncon({E, r}, {[-1 -2 1 2], [1 2]}), tol);
checkl = ArrayIsEqual(l, ncon({l, E}, {[1 2], [2 1 -2 -1]}), tol);
checktr = ArrayIsEqual(1, ncon({l, r},{[1 2], [2 1]}), tol);


%% Calculate left and right orthonormal gauge transform

% first option: taking square root of left fixed point; less accurate option
[V, S] = eig(l);
L1 = V * sqrt(S) * V';
[V, S] = eig(r);
R1 = V * sqrt(S) * V';

AL1 = ncon({L1, A, inv(L1)}, {[-1 1], [1 -2 2], [2 -3]}); % MPS tensor in left canonical form
AR1 = ncon({inv(R1), A, R1}, {[-1 1], [1 -2 2], [2 -3]}); % MPS tensor in right canonical form
checkL1 = ArrayIsEqual(eye(D), ncon({AL1, conj(AL1)}, {[1 2 -1], [1 2 -2]}), 1e-9); % check AL: ok, but requires pretty high tolerance
checkR1 = ArrayIsEqual(eye(D), ncon({AR1, conj(AR1)}, {[-1 1 2], [-2 1 2]}), 1e-9); % check AR: ok, but requires pretty high tolerance


% second option: through iterative QR decompositions; more accurate option

% left orthonormalization -> implemented in function LeftOrthonormalize
L = randcomplex(D, D); % initialize random matrix
L = L / ArrayNorm(L); % normalize
flag = true;
while flag
    [AL2, Lprime] = qrpos(reshape(ncon({L, A}, {[-1 1], [1 -2 -3]}), [D*d, D]));
    lambda = ArrayNorm(Lprime);    Lprime = Lprime / lambda;
    if ArrayIsEqual(L, Lprime, 1e-14)
        flag = false;
    else
        L = Lprime;
    end
end
L2 = Lprime;
AL2 = reshape(AL2, [D d D]);

% right orthonormalization -> implemented in function RightOrthonormalize
R = randcomplex(D, D); % initialize random matrix
R = R / ArrayNorm(R); % normalize
flag = true;
while flag
    [AR2, Rprime] = lq(reshape(ncon({A, R}, {[-1 -2 1], [1 -3]}), [D, d*D]));
    Rprime = Rprime / ArrayNorm(Rprime);
    if ArrayIsEqual(R, Rprime, 1e-14)
        flag = false;
    else
        R = Rprime;
    end
end
R2 = Rprime;
AR2 = reshape(AR2, [D d D]);

checkL2 = ArrayIsEqual(eye(D), ncon({AL2, conj(AL2)}, {[1 2 -1], [1 2 -2]}), tol); % check AL: ok, with lower tolerance
checkL2_extra = ArrayIsEqual(AL2, ncon({L2, A, inv(L2)}, {[-1 1], [1 -2 2], [2, -3]}), tol);
checkR2 = ArrayIsEqual(eye(D), ncon({AR2, conj(AR2)}, {[-1 1 2], [-2 1 2]}), tol); % check AR: ok, with lower tolerance
checkR2_extra = ArrayIsEqual(AR2, ncon({inv(R2), A, R2}, {[-1 1], [1 -2 2], [2, -3]}), tol);

% didn't do the hybrid method using an extra Arnoldi solver step from lecture notes, is this necessary?

% why would we need the lambda from algorithm 1 box in the lecture notes? Isn't this just always going to be one if the procedure converges?

% use one result for the remainder
L = L2; R = R2; AR = AR2; AL = AL2;

% some extra checks, just for my own sanity; seems fine now
l2 = L'*L;
r2 = R*R';
tr2 = ncon({l2, r2},{[1 2], [2 1]});
l2 = l2 / sqrt(tr2); r2 = r2 / sqrt(tr2);
checkr2 = ArrayIsEqual(r2, ncon({E, r2}, {[-1 -2 1 2], [1 2]}), tol);
checkl2 = ArrayIsEqual(l2, ncon({l2, E}, {[1 2], [2 1 -2 -1]}), tol);
checktr2 = ArrayIsEqual(1, ncon({l2, r2},{[1 2], [2 1]}), tol);


%% Go to mixed gauge

% -> implemented in function MixedCanonical (but without truncation)
C = L * R;
checkC1 = ArrayIsEqual(ncon({AL, C}, {[-1 -2 1], [1 -3]}), ncon({C, AR}, {[-1 1], [1 -2 -3]}), tol); % check property (16) of the C-matrix

% diagonalize C using SVD
[U, C, V] = svd(C);

% normalize center matrix
nrm = trace(C * C');
C = C / sqrt(nrm);

% apply truncation if desired, default no truncation, t = 0
t = 0;
if t
    % if t > 0, keep t highest values in C and their according unitaries U and V
    U = U(:,1:t);
    C = C(1:t,1:t);
    V = V(:,1:t);
    AL = ncon({U', AL, U}, {[-1 1], [1 -2 2], [2 -3]});
    AR = ncon({V', AR, V}, {[-1 1], [1 -2 2], [2 -3]});
    AC = ncon({AL, C}, {[-1 -2 1], [1 -3]});
    
    checkL = ArrayIsEqual(eye(t), ncon({AL, conj(AL)}, {[1 2 -1], [1 2 -2]}), tol); % doesn't check out after truncating: need to normalize again?
    checkR = ArrayIsEqual(eye(t), ncon({AR, conj(AR)}, {[-1 1 2], [-2 1 2]}), tol); % doesn't check out after truncating: need to normalize again?
    checkC2 = ArrayIsEqual(ncon({AL, C}, {[-1 -2 1], [1 -3]}), ncon({C, AR}, {[-1 1], [1 -2 -3]}), tol); % recheck property (16) of the C-matrix
    
    % entanglement entropy
    S = -sum(C^2*diag(log(C^2)));
    
else
    AL = ncon({U', AL, U}, {[-1 1], [1 -2 2], [2 -3]});
    AR = ncon({V', AR, V}, {[-1 1], [1 -2 2], [2 -3]});
    AC = ncon({AL, C}, {[-1 -2 1], [1 -3]});
    checkL = ArrayIsEqual(eye(D), ncon({AL, conj(AL)}, {[1 2 -1], [1 2 -2]}), tol); % check AL again, just to see if I daggered the right thing just now
    checkR = ArrayIsEqual(eye(D), ncon({AR, conj(AR)}, {[-1 1 2], [-2 1 2]}), tol); % check AR again, just to see if I daggered the right thing just now
    checkC2 = ArrayIsEqual(ncon({AL, C}, {[-1 -2 1], [1 -3]}), ncon({C, AR}, {[-1 1], [1 -2 -3]}), tol); % recheck property (16) of the C-matrix
    
    % entanglement entropy
    S = -sum(C^2*diag(log(C^2)));
end


%% Expectation values of single- and two-site operators

% random one- and two-site operators to test with
O1 = randcomplex(d, d);
O2 = randcomplex(d, d, d, d);

% expectation value of single-site operator
% uniform gauge -> implemented in function ExpvOneBodyUniform
expv1_1 = ncon({l, r, A, conj(A), O1}, {...
    [4 1],...
    [3 6],...
    [1 2 3],...
    [4 5 6],...
    [2 5]});
% mixed gauge -> implemented in function ExpvOneBodyMixed
expv1_2 = ncon({AC, conj(AC), O1}, {...
    [1 2 3],...
    [1 4 3],...
    [2 4]}, [2 1 3 4]);
checkExpv1 = ArrayIsEqual(expv1_1, expv1_2, tol);

% expectation value of two-site operator
% uniform gauge -> implemented in function ExpvTwoBodyUniform
expv2_1 = ncon({l, r, A, A, conj(A), conj(A), O2}, {...
    [6 1],...
    [5 10],...
    [1 2 3],...
    [3 4 5],...
    [6 7 8],...
    [8 9 10],...
    [2 4 7 9]});
% mixed gauge -> implemented in function ExpvTwoBodyMixed
expv2_2 = ncon({AL, AC, conj(AL), conj(AC), O2}, {...
    [1 2 3],...
    [3 4 5],...
    [1 6 7],...
    [7 8 5],...
    [2 4 6 8]}, [3 2 4 1 6 5 8 7]);
checkExpv2 = ArrayIsEqual(expv2_1, expv2_2, tol);


%% Variational optimization of spin-1 Heisenberg Hamiltonian with gradient descent in uniform gauge

% coupling strengths
Jx = -1; Jy = -1; Jz = -1; hz = 0; % Heisenberg antiferromagnet
% Heisenberg Hamiltonian
h = HeisenbergHamiltonian(Jx, Jy, Jz, hz);

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
    e
    htilde = h - e * ncon({eye(d), eye(d)}, {[-1 -3], [-2 -4]}); % regularized energy density
    Rh = RightEnvMixed(AR, C, htilde, delta);
    Lh = LeftEnvMixed(AL, C, htilde, delta);
    [ACprime, Cprime] = CalculateNewCenter(AL, AR, AC, C, Rh, Lh, htilde, delta);
    [ALprime, ARprime, ACprime, Cprime] = MinAcC(ACprime, Cprime);
    delta = ArrayNorm(H_AC(AC, AL, AR, Rh, Lh, htilde) - ncon({AL, H_C(C, AL, AR, Rh, Lh, htilde)}, {[-1 -2 1], [1 -3]})); % calculate error using new or old AL, AR, Rh, Lh? now using old...
    delta
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

%% VUMPS for 2d classical Ising model

tol = 1e-6;

beta = 0.5; % 2.269
J = 1;

D = 12;
d = 2;
A = randcomplex(D, d, D); % MPS tensor

O = IsingO(beta, J);

% algorithm 8 for finding MPS fixed point of a given MPO
[AL, AR, AC, C] = MixedCanonical(A); % go to mixed gauge
flag = true;
delta = 1e-4;
i = 0;
while flag
    [lambda, FL] = FixedPointLeft(AL, O, delta);
    [tst, FR] = FixedPointRight(AR, O, delta);
    FL = FL / OverlapFixedPoints(FL, FR, C);
    [ACprime, Cprime] = CalculateNewCenter2D(AC, C, FL, FR, O, lambda, delta);
    [ALprime, ARprime, ACprime, Cprime] = MinAcC(ACprime, Cprime);
    delta = ArrayNorm(OAC(AC, FL, FR, O, lambda) - ncon({AL, OC(C, FL, FR)}, {[-1 -2 1], [1 -3]})); % calculate error using new or old AL, AR, Rh, Lh? now using old...
    delta
    AL = ALprime; AR = ARprime; AC = ACprime; C = Cprime; % update
    i = i+1;
    if delta < tol
        flag = false;
    end
end
fprintf('Iterations needed: %i\n', i)
freeEnergy = -log(lambda)/beta;
[~, freeEnergyExact, ~] = isingExact(J, beta);
% check free energy
freeEnergy
freeEnergyExact
abs(freeEnergyExact - freeEnergy)/abs(freeEnergy) < 1e-5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% function definitions

%% normalizing/gauging MPS

function E = CreateTransfer(A)
    E = ncon({A, conj(A)}, {[-1 1 -2], [-3 1 -4]});
end

function [A, l, r] = NormalizeMPS(A)
    D = size(A, 1);
    % calculate left and right fixed points
    handleR = @(v) reshape(ncon({A, conj(A), reshape(v, [D D])}, {[-1 2 1], [-2 2 3], [1 3]}), [], 1);
    handleL = @(v) reshape(ncon({A, conj(A), reshape(v, [D D])}, {[1 2 -2], [3 2 -1], [3 1]}), [], 1);
    [r, lambda] = eigs(handleR, D^2, 1); % right eigenvector
    [l, ~] = eigs(handleL, D^2, 1); % left eigenvector
    r = reshape(r, [D D]);
    l = reshape(l, [D D]);
    % normalize MPS tensor
    A = A / sqrt(lambda);
    % normalize fixed points
    tr = ncon({l, r},{[1 2], [2 1]});
    l = l / sqrt(tr); r = r / sqrt(tr);
end

function [AL, L, lambda] = LeftOrthonormalize(A, L0, tol)
    D = size(A, 1); d = size(A, 2);
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
        [AL, Lprime] = qrpos(reshape(ncon({L, A}, {[-1 1], [1 -2 -3]}), [D*d, D]));
        lambda = ArrayNorm(Lprime);    Lprime = Lprime / lambda;
        if ArrayIsEqual(L, Lprime, tol)
            flag = false;
        else
            L = Lprime;
        end
    end
    L = Lprime;
    AL = reshape(AL, [D d D]);
end

function [AR, R, lambda] = RightOrthonormalize(A, R0, tol)
    D = size(A, 1); d = size(A, 2);
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
        [AR, Rprime] = lq(reshape(ncon({A, R}, {[-1 -2 1], [1 -3]}), [D, d*D]));
        lambda = ArrayNorm(Rprime);    Rprime = Rprime / lambda;
        if ArrayIsEqual(R, Rprime, tol)
            flag = false;
        else
            R = Rprime;
        end
    end
    R = Rprime;
    AR = reshape(AR, [D d D]);
end

function [AL, AR, AC, C] = MixedCanonical(A, tol)
    if nargin < 2
        tol = 1e-12;
    end
    D = size(A, 1);
    R0 = randcomplex(D, D); L0 = randcomplex(D, D); % initialize random matrices
    [AL, L, ~] = LeftOrthonormalize(A, L0, tol);
    [AR, R, ~] = RightOrthonormalize(A, R0, tol);
    [U, C, V] = svd(L * R);
    % normalize center matrix
    nrm = trace(C * C');
    C = C / sqrt(nrm);
    % compute MPS tensors
    AL = ncon({U', AL, U}, {[-1 1], [1 -2 2], [2 -3]});
    AR = ncon({V', AR, V}, {[-1 1], [1 -2 2], [2 -3]});
    AC = ncon({AL, C}, {[-1 -2 1], [1 -3]});
end

%% expectation values of on- and two-site operators in uniform and mixed gauge

function expv = ExpvOneSiteUniform(A, l, r, O)
    expv = ncon({l, r, A, conj(A), O}, {...
        [4 1],...
        [3 6],...
        [1 2 3],...
        [4 5 6],...
        [2 5]});
end

function expv = ExpvOneSiteMixed(AC, O)
    expv = ncon({AC, conj(AC), O}, {...
        [1 2 3],...
        [1 4 3],...
        [2 4]}, [2 1 3 4]);
end

function expv = ExpvTwoSiteUniform(A, l, r, O)
    expv = ncon({l, r, A, A, conj(A), conj(A), O}, {...
        [6 1],...
        [5 10],...
        [1 2 3],...
        [3 4 5],...
        [6 7 8],...
        [8 9 10],...
        [2 4 7 9]});
end

function expv = ExpvTwoSiteMixed(AC, AL, O)
    expv = ncon({AL, AC, conj(AL), conj(AC), O}, {...
        [1 2 3],...
        [3 4 5],...
        [1 6 7],...
        [7 8 5],...
        [2 4 6 8]}, [3 2 4 1 6 5 8 7]);
end

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

%% VUMPS algorithm for 2-dimensional classical partition function

function out = delt(n, d)
    out = zeros(repmat(d, 1, n));
    for i = 1:d
        sbs = num2cell(repmat(i, 1, n));
        out(sbs{:}) = 1;
    end
end

function O = IsingO(beta, J)
    c = sqrt(cosh(beta*J)); s = sqrt(sinh(beta*J));
    Q_sqrt = 1/sqrt(2) * [c+s, c-s; c-s, c+s];
    O = ncon({Q_sqrt, Q_sqrt, Q_sqrt, Q_sqrt, delt(4, 2)}, {[-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]});
end

function [magnetization,free,energy]=isingExact(J,beta)
    theta=0:1e-6:pi/2;
    x=2*sinh(2*J*beta)/cosh(2*J*beta)^2;
    if 1-(sinh(2*J*beta))^(-4)>0
        magnetization=(1-(sinh(2*J*beta))^(-4))^(1/8);
    else
        magnetization=0;
    end
    free=-1/beta*(log(2*cosh(2*J*beta))+1/pi*trapz(theta,log(1/2*(1+sqrt(1-x^2*sin(theta).^2)))));
    K=trapz(theta,1./sqrt(1-x^2*sin(theta).^2));
    energy=-J*cosh(2*J*beta)/sinh(2*J*beta)*(1+2/pi*(2*tanh(2*J*beta)^2-1)*K);
end

function [lambda, FL] = FixedPointLeft(AL, O, delta)
    D = size(AL, 1); d = size(AL, 2);
    handleL = @(v) reshape(ncon({reshape(v, [D d D]), AL, conj(AL), O}, {[5, 3, 1], [1, 2, -3], [5 4 -1], [3 2 -2 4]}), [], 1);
    [FL, lambda] = eigs(handleL, D^2*d, 1, 'largestabs', 'Tolerance', delta/10); % left eigenvector
    FL = reshape(FL, [D d D]);
end

function [lambda, FR] = FixedPointRight(AR, O, delta)
    D = size(AR, 1); d = size(AR, 2);
    handleR = @(v) reshape(ncon({reshape(v, [D d D]), AR, conj(AR), O}, {[1, 3, 5], [-1, 2, 1], [-3, 4, 5], [-2, 2, 3, 4]}), [], 1);
    [FR, lambda] = eigs(handleR, D^2*d, 1, 'largestabs', 'Tolerance', delta/10); % right eigenvector
    FR = reshape(FR, [D d D]);
end

function overl = OverlapFixedPoints(FL, FR, C)
    overl = ncon({FL, FR, C, conj(C)}, {[1, 3, 2], [5, 3, 4], [2, 5], [1, 4]});
end

function Xprime = OAC(X, FL, FR, O, lambda)
    Xprime = ncon({FL, FR, X, O}, {[-1, 2, 1], [4, 5, -3], [1, 3, 4], [2, 3, 5, -2]}) / lambda;
end

function Xprime = OC(X, FL, FR)
    Xprime = ncon({FL, FR, X}, {[-1, 3, 1], [2, 3, -2], [1, 2]});
end

function [ACprime, Cprime] = CalculateNewCenter2D(AC, C, FL, FR, O, lambda, delta)
    D = size(AC, 1); d = size(AC, 2);
    % compute action of maps (256) and (257) in the notes and pour this into function handle for eigs
    handleAC = @(X) reshape(OAC(reshape(X, [D d D]), FL, FR, O, lambda), [], 1);
    handleC = @(X) reshape(OC(reshape(X, [D D]), FL, FR), [], 1);
    % solve eigenvalue problem using 'largest magnitude' option
    [ACprime, ~] = eigs(handleAC, D^2*d, 1, 'largestabs', 'Tolerance', delta/10, 'StartVector', reshape(AC, [], 1)); % variable tolerance
    ACprime = reshape(ACprime, [D d D]);
    [Cprime, ~] = eigs(handleC, D^2, 1, 'largestabs', 'Tolerance', delta/10, 'StartVector', reshape(C, [], 1)); % variable tolerance
    Cprime = reshape(Cprime, [D D]);
end
