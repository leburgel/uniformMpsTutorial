% Matlab script for chapter 1 of Bad Honnef tutorial on "Tangent space
% methods for Tangent-space methods for uniform matrix product states",
% based on the lecture notes: https://arxiv.org/abs/1810.07006
% 
% Detailed explanations of all the different steps can be found in the
% python notebooks for the different chapters. These files provide a canvas
% for a MATLAB implementation that mirrors the contents of the python
% notebooks

%% 1. Matrix product states in the thermodynamic limit

% Unlike the notebooks, where function definitions and corresponding checks
% are constructed in sequence, here all checks are placed at the start of
% the script, while all function definitions must be given at the bottom of
% the script


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHECKS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 1.1 Normalisation

d = 3;
D = 5;

% initializing a random MPS:
A = createMPS(D, d);

assert(isequal(size(A), [D, d, D]), 'Generated MPS tensor has incorrect shape.')
assert(~isreal(A), 'MPS tensor should have complex values')


% normalizing an MPS through naive diagonalization of the transfer matrix:
A = normalizeMPSNaive(A);
[l, r] = fixedPointsNaive(A);

assert(ArrayIsEqual(l, l', 1e-12), 'left fixed point should be hermitian!')
assert(ArrayIsEqual(r, r', 1e-12), 'left fixed point should be hermitian!')

assert(ArrayIsEqual(l, ncon({A, l, conj(A)}, {[1, 2, -2], [3, 1], [3, 2, -1]}), 1e-12), 'l should be a left fixed point!')
assert(ArrayIsEqual(r, ncon({A, r, conj(A)}, {[-1, 2, 1], [1, 3], [-2, 2, 3]}), 1e-12), 'r should be a right fixed point!')
assert(abs(trace(l*r) - 1) < 1e-12, 'Left and right fixed points should be trace normalized!')


%% 1.2 Gauge fixing

% left and right orthonormalisation through taking square root of fixed points

[L, Al] = leftOrthonormalizeNaive(A, l);
[R, Ar] = rightOrthonormalizeNaive(A, r);

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

% normalizing an MPS through action of transfer matrix on a left and right matrix as a function handle:
A = normalizeMPS(A);
[l, r] = fixedPoints(A);

assert(ArrayIsEqual(l, l', 1e-12), 'left fixed point should be hermitian!')
assert(ArrayIsEqual(r, r', 1e-12), 'left fixed point should be hermitian!')

assert(ArrayIsEqual(l, ncon({A, l, conj(A)}, {[1, 2, -2], [3, 1], [3, 2, -1]}), 1e-12), 'l should be a left fixed point!')
assert(ArrayIsEqual(r, ncon({A, r, conj(A)}, {[-1, 2, 1], [1, 3], [-2, 2, 3]}), 1e-12), 'r should be a right fixed point!')
assert(abs(trace(l*r) - 1) < 1e-12, 'Left and right fixed points should be trace normalized!')


% gauging an MPS through iterative QR decompositions:
[Al, Ac, Ar, C] = mixedCanonical(A);
% [R, Ar] = rightOrthonormalize(A);
% [L, Al] = leftOrthonormalize(A);

assert(ArrayIsEqual(eye(D), ncon({Ar, conj(Ar)}, {[-1 1 2], [-2 1 2]}), 1e-12), 'Ar not in right-orthonormal form')
assert(ArrayIsEqual(eye(D), ncon({Al, conj(Al)}, {[1 2 -2], [1 2 -1]}), 1e-12), 'Al not in left-orthonormal form')
LHS = ncon({Al, C}, {[-1, -2, 1], [1, -3]});
RHS = ncon({C, Ar}, {[-1, 1], [1, -2, -3]});
assert(ArrayIsEqual(LHS, RHS, 1e-12) && ArrayIsEqual(RHS, Ac, 1e-12), 'Something went wrong in gauging the MPS')


%% 1.5 Computing expectation values
A = createMPS(D, d);
A = normalizeMPS(A);
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

%% 1.1 Normalisation

function A =  createMPS(D, d)
    % Returns a random complex MPS tensor.
    %     Parameters
    %     ----------
    %     D : int
    %         Bond dimension for MPS.
    %     d : int
    %         Physical dimension for MPS.
    %
    %     Returns
    %     -------
    %     A : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.

end


function E = createTransfermatrix(A)
    % Form the transfermatrix of an MPS.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     E : array(D, D, D, D)
    %         Transfermatrix with 4 legs,
    %         ordered topLeft-bottomLeft-topRight-bottomRight.
    
    % given as an example
    
    E = ncon({A, conj(A)}, {[-1 1 -3], [-2 1 -4]});
end


function Anew = normalizeMPSNaive(A)
    % Normalize an MPS tensor.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     Anew : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Complexity
    %     ----------
    %     O(D ** 6) algorithm,
    %         directly diagonalizing (D ** 2, D ** 2) matrix.

end


function l = leftFixedPointNaive(A)
    % Find left fixed point.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     l : array(D, D)
    %         left fixed point with 2 legs,
    %         ordered bottom-top.
    %
    %     Complexity
    %     ----------
    %     O(D ** 6) algorithm,
    %         diagonalizing (D ** 2, D ** 2) matrix.

end


function r = rightFixedPointNaive(A)
    % Find left fixed point.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     l : array(D, D)
    %         left fixed point with 2 legs,
    %         ordered bottom-top.
    %
    %     Complexity
    %     ----------
    %     O(D ** 6) algorithm,
    %         diagonalizing (D ** 2, D ** 2) matrix.

end


function [l, r] = fixedPointsNaive(A)
    % Find normalized fixed points.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     l : array(D, D)
    %         left fixed point with 2 legs,
    %         ordered bottom-top.
    %     r : array(D, D)
    %         right fixed point with 2 legs,
    %         ordered top-bottom.
    %
    %     Complexity
    %     ----------
    %     O(D ** 6) algorithm,
    %         diagonalizing (D ** 2, D ** 2) matrix

end


%% 1.2 Gauge fixing


function [L, Al] = leftOrthonormalizeNaive(A, l)
    % Transform A to left-orthonormal gauge.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     L : array(D, D)
    %         left gauge with 2 legs,
    %         ordered left-right.
    %     Al : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal
    %
    %     Complexity
    %     ----------
    %     O(D ** 6) algorithm,
    %         diagonalizing (D ** 2, D ** 2) matrix

end


function [R, Ar] = rightOrthonormalizeNaive(A, r)
    % Transform A to right-orthonormal gauge.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     R : array(D, D)
    %         right gauge with 2 legs,
    %         ordered left-right.
    %     Ar : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal
    %
    %     Complexity
    %     ----------
    %     O(D ** 6) algorithm,
    %         diagonalizing (D ** 2, D ** 2) dmatrix

end


function [Al, Ac, Ar, C] = mixedCanonicalNaive(A)
    % Bring MPS tensor into mixed gauge, such that -Al-C- = -C-Ar- = Ac.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     Al : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal.
    %     Ac : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     Ar : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right orthonormal.
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right,
    %         diagonal.
    %
    %     Complexity
    %     ----------
    %     O(D ** 6) algorithm,
    %         diagonalisation of (D ** 2, D ** 2) matrix

end


%% 1.3 Truncation of a uniform MPS
function [AlTilde, AcTilde, ArTilde, CTilde] = truncateMPS(A, Dtrunc)
    % Truncate an MPS to a lower bond dimension.
    %
    %     Parameters
    %     ----------
    %     A : np.array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     Dtrunc : int
    %         lower bond dimension
    %
    %     Returns
    %     -------
    %     AlTilde : array(Dtrunc, d, Dtrunc)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal.
    %     AcTilde : array(Dtrunc, d, Dtrunc)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     ArTilde : array(Dtrunc, d, Dtrunc)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right orthonormal.
    %     CTilde : array(Dtrunc, Dtrunc)
    %         Center gauge with 2 legs,
    %         ordered left-right,
    %         diagonal.

end


%% 1.4 Algorithms for finding canonical forms


function Anew =  normalizeMPS(A)
    % Normalize an MPS tensor.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     Anew : np.array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Complexity
    %     ----------
    %     O(D ** 3) algorithm,
    %         D ** 3 contraction for transfer matrix handle.

end


function l = leftFixedPoint(A)
    % Find left fixed point.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     l : array(D, D)
    %         left fixed point with 2 legs,
    %         ordered bottom-top.
    %
    %     Complexity
    %     ----------
    %     O(D ** 3) algorithm,
    %          D ** 3 contraction for transfer matrix handle.

end


function r = rightFixedPoint(A)
    % Find left fixed point.
    %
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %
    %     Returns
    %     -------
    %     l : array(D, D)
    %         left fixed point with 2 legs,
    %         ordered bottom-top.
    %
    %     Complexity
    %     ----------
    %     O(D ** 3) algorithm,
    %          D ** 3 contraction for transfer matrix handle.

end


function [l, r] = fixedPoints(A)
    % Find normalized fixed points.
    % 
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    % 
    %     Returns
    %     -------
    %     l : array(D, D)
    %         left fixed point with 2 legs,
    %         ordered bottom-top.
    %     r : array(D, D)
    %         right fixed point with 2 legs,
    %         ordered top-bottom.
    % 
    %     Complexity
    %     ----------
    %     O(D ** 3) algorithm,
    %          D ** 3 contraction for transfer matrix handle.
    
end


% rqPos: already implemented as function 'lq' in the folder 'AuxiliaryFunctions'


function [R, Ar] = rightOrthonormalize(A, R0, tol, maxIter)
    % Transform A to right-orthonormal gauge.
    % 
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     R0 : array(D, D), optional
    %         Right gauge matrix,
    %         initial guess.
    %     tol : float, optional
    %         convergence criterium,
    %         norm(R - Rnew) < tol.
    %     maxIter : int
    %         maximum amount of iterations.
    % 
    %     Returns
    %     -------
    %     R : array(D, D)
    %         right gauge with 2 legs,
    %         ordered left-right.
    %     Ar : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right-orthonormal

end


% qrPos: already implemented as function 'qrpos' in the folder 'AuxiliaryFunctions'


function [L, Al] = leftOrthonormalize(A, L0, tol, maxIter)
    % Transform A to left-orthonormal gauge.
    % 
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     L0 : array(D, D), optional
    %         Left gauge matrix,
    %         initial guess.
    %     tol : float, optional
    %         convergence criterium,
    %         norm(R - Rnew) < tol.
    %     maxIter : int
    %         maximum amount of iterations.
    % 
    %     Returns
    %     -------
    %     L : array(D, D)
    %         left gauge with 2 legs,
    %         ordered left-right.
    %     Al : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left-orthonormal
            
end


function [Al, Ac, Ar, C] = mixedCanonical(A, tol)
    % Bring MPS tensor into mixed gauge, such that -Al-C- = -C-Ar- = Ac.
    % 
    %     Parameters
    %     ----------
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    % 
    %     Returns
    %     -------
    %     Al : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal.
    %     Ac : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     Ar : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right orthonormal.
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right,
    %         diagonal.
    % 
    %     Complexity
    %     ----------
    %     O(D ** 3) algorithm.
    
end


%% 1.5 Computing expectation values 

function o = expVal1Uniform(O, A, l, r)
    % Calculate the expectation value of a 1-site operator in uniform gauge.
    % 
    %     Parameters
    %     ----------
    %     O : array(d, d)
    %         single-site operator.
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalized.
    %     r : array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalized.
    % 
    %     Returns
    %     -------
    %     o : complex float
    %         expectation value of O.
    
    % given as an example:
    
    if nargin < 4
        [l, r] = fixedPoints(A);
    end
    
    o = ncon({l, r, A, conj(A), O}, {[4 1], [3 6], [1 2 3], [4 5 6], [2 5]});
end

function o = expVal1Mixed(O, AC)
    % Calculate the expectation value of a 1-site operator in mixed gauge.
    % 
    %     Parameters
    %     ----------
    %     O : array(d, d)
    %         single-site operator.
    %     Ac : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauged.
    % 
    %     Returns
    %     -------
    %     o : complex float
    %         expectation value of O.

    o = ncon({AC, conj(AC), O}, {[1 2 3], [1 4 3], [2 4]}, [2 1 3 4]);
end

function o = expVal2Uniform(O, A, l, r)
    % Calculate the expectation value of a 2-site operator in uniform gauge.
    % 
    %     Parameters
    %     ----------
    %     O : array(d, d, d, d)
    %         two-site operator,
    %         ordered topLeft-topRight-bottomLeft-bottomRight.
    %     A : array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right.
    %     l : array(D, D), optional
    %         left fixed point of transfermatrix,
    %         normalized.
    %     r : array(D, D), optional
    %         right fixed point of transfermatrix,
    %         normalized.
    % 
    %     Returns
    %     -------
    %     o : complex float
    %         expectation value of O.

    o = ncon({l, r, A, A, conj(A), conj(A), O}, {[6 1], [5 10], [1 2 3], [3 4 5], [6 7 8], [8 9 10], [2 4 7 9]});
end

function o = expVal2Mixed(O, Al, Ac)
    % Calculate the expectation value of a 2-site operator in mixed gauge.
    % 
    %     Parameters
    %     ----------
    %     O : array(d, d, d, d)
    %         two-site operator,
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
    %     o : complex float
    %         expectation value of O.

    o = ncon({Al, Ac, conj(Al), conj(Ac), O}, { [1 2 3], [3 4 5], [1 6 7], [7 8 5], [2 4 6 8]}, [3 2 4 1 6 5 8 7]);
end
