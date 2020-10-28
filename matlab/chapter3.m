% Matlab script for chapter 3 of Bad Honnef tutorial on "Tangent space
% methods for Tangent-space methods for uniform matrix product states",
% based on the lecture notes: https://arxiv.org/abs/1810.07006
% 
% Detailed explanations of all the different steps can be found in the
% python notebooks for the different chapters. These files provide a canvas
% for a MATLAB implementation that mirrors the contents of the python
% notebooks

%% 3. VUMPS for MPO's, two-dimensional partition functions and PEPS

% Unlike the notebooks, where function definitions and corresponding checks
% are constructed in sequence, here all checks and demonstrations are
% placed at the start of the script, while all function definitions must
% be given at the bottom of the script


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DEMONSTRATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3.2 The 2d classical Ising model


% test vumps for 2d ising partition function
D = 12;
d = 2;

fprintf('Bond dimension: D =%i\n', D)

beta = 1; % critical point: 0.440686793509772
O = isingO(beta, 1);

A = createMPS(D, d);
tol = 1e-5;

tic
[lam, Al, Ac, Ar, C, Fl, Fr] = vumpsMPO(O, D, A, tol);
time = toc;
fprintf('Time for VUMPS optimization: %f s\n', time)
freeEnergy = -log(lam) / beta;
[~, freeEnergyExact, ~] = isingExact(beta, 1);
fprintf('Computed free energy: %.10f\n', freeEnergy)
fprintf('Exact free energy: %.10f\n', freeEnergyExact)



% compute magnetization and free energy curves
D = 12;
d = 2;
J = 1;

fprintf('\nBond dimension: D =%i\n', D)
Al = createMPS(D, d);
% optimization parameters
tol = 1e-5;

Ts = linspace(1., 3.4, 100);
magnetizations = [];
magnetizationsExact = [];
freeEnergies = [];
freeEnergiesExact = [];

for T = Ts
    beta = 1/T;
    O = isingO(beta, J);
    fprintf('T=%f\n', T)
    [lam, Al, Ac, Ar, C, Fl, Fr] = vumpsMPO(O, D, Al, tol);
    magnetizations = [magnetizations, abs(isingMagnetization(beta, J, Ac, Fl, Fr)/isingZ(beta, J, Ac, Fl, Fr))];
    [mEx, fEx, ~] = isingExact(beta, J);
    magnetizationsExact = [magnetizationsExact, mEx];
    freeEnergies = [freeEnergies, -log(lam)/beta];
    freeEnergiesExact = [freeEnergiesExact, fEx];
end

figure
scatter(Ts, magnetizations, 'x')
plot(Ts, magnetizationsExact)
title('Magnetization as a function of the temperature')

figure
scatter(Ts, freeEnergies, 'x')
plot(Ts, freeEnergiesExact)
title('Free energy as a function of the temperature')




%% 3.3 VUMPS for PEPS


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION DEFINITIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 3.1 VUMPS for MPO's

function [lam, Fl] = leftFixedPointMPO(O, Al, tol)
    % Computes the left fixed point (250).
    % 
    %     Parameters
    %     ----------
    %     O : array (d, d, d, d)
    %         MPO tensor,
    %         ordered left-top-right-bottom.
    %     Al : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         left-orthonormal.
    %     tol : float, optional
    %         current tolerance
    % 
    %     Returns
    %     -------
    %     lam : float
    %         Leading left eigenvalue.
    %     Fl : array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    
    % given as an example
    
    D = size(Al, 1);
    d = size(Al, 2);
    
    % construct handle for the action of the relevant operator
    transferLeftHandleMPO = @(v) reshape(ncon({reshape(v, [D d D]), Al, conj(Al), O}, {[5, 3, 1], [1, 2, -3], [5 4 -1], [3 2 -2 4]}), [], 1);
    [Fl, lam] = eigs(transferLeftHandleMPO, D^2*d, 1, 'largestabs', 'Tolerance', tol); % left eigenvector
    Fl = reshape(Fl, [D d D]);
end


function [lam, Fr] = rightFixedPointMPO(O, Ar, tol)
    % Computes the right fixed point (250).
    % 
    %     Parameters
    %     ----------
    %     O : array (d, d, d, d)
    %         MPO tensor,
    %         ordered left-top-right-bottom.
    %     Ar : array (D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         right-orthonormal.
    %     tol : float, optional
    %         current tolerance
    % 
    %     Returns
    %     -------
    %     lam : float
    %         Leading right eigenvalue.
    %     Fr : array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.

end


function overlap = overlapFixedPointsMPO(Fl, Fr, C)
    % Performs the contraction that gives the overlap of the fixed points (251).
    % 
    %     Parameters
    %     ----------
    %     Fl : array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    %     Fr : array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.
    %     C : array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right.
    % 
    %     Returns
    %     -------
    %     overlap : float
    %         Overlap of the fixed points.
    
    % given
    
    overlap = ncon({Fl, Fr, C, conj(C)}, {[1, 3, 2], [5, 3, 4], [2, 5], [1, 4]});
end


function Xnew = O_Ac(X, O, Fl, Fr, lam)
    % Action of the map (256) on a given tensor.
    % 
    %     Parameters
    %     ----------
    %     X : array(D, d, D)
    %         Tensor of size (D, d, D)
    %     O : np.array (d, d, d, d)
    %         MPO tensor,
    %         ordered left-top-right-bottom.
    %     Fl : array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    %     Fr : array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.
    %     lam : float
    %         Leading eigenvalue.
    % 
    %     Returns
    %     -------
    %     Xnew : array(D, d, D)
    %         Result of the action of O_Ac on the tensor X.
    
    % given as an example
    
    Xnew = ncon({Fl, Fr, X, O}, {[-1, 2, 1], [4, 5, -3], [1, 3, 4], [2, 3, 5, -2]}) / lam;
end


function Xnew = O_C(X, Fl, Fr)
    % Action of the map (257) on a given tensor.
    % 
    %     Parameters
    %     ----------
    %     X : array(D, D)
    %         Tensor of size (D, D)
    %     Fl : np.array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    %     Fr : array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.
    % 
    %     Returns
    %     -------
    %     Xnew : array(D, d, D)
    %         Result of the action of O_C on the tensor X.
    
end


function [AcTilde, CTilde] = calcNewCenterMPO(O, Ac, C, Fl, Fr, lam, tol)
    % Find new guess for Ac and C as fixed points of the maps O_Ac and O_C.
    % 
    %     Parameters
    %     ----------
    %     O : np.array (d, d, d, d)
    %         MPO tensor,
    %         ordered left-top-right-bottom.
    %     Ac : np.array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     C : np.array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right.
    %     Fl : np.array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    %     Fr : np.array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.
    %     lam : float
    %         Leading eigenvalue.
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

end


function [lam, Al, Ac, Ar, C, Fl, Fr] = vumpsMPO(O, D, A0, tol)
    % Find the fixed point MPS of a given MPO using VUMPS.
    % 
    %     Parameters
    %     ----------
    %     O : np.array (d, d, d, d)
    %         MPO tensor,
    %         ordered left-top-right-bottom.
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
    %     lam : float
    %         Leading eigenvalue.
    %     Al : np.array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         left orthonormal.
    %     Ar : np.array(D, d, D)
    %         MPS tensor zith 3 legs,
    %         ordered left-bottom-right,
    %         right orthonormal.
    %     Ac : np.array(D, d, D)
    %         MPS tensor with 3 legs,
    %         ordered left-bottom-right,
    %         center gauge.
    %     C : np.array(D, D)
    %         Center gauge with 2 legs,
    %         ordered left-right.
    %     Fl : np.array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    %     Fr : np.array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.
            
end


%% 3.2 The 2d classical Ising model


function O = isingO(beta, J)
    % Gives the MPO tensor corresponding to the partition function of the 2d 
    % classical Ising model at a given temperature and coupling, obtained by
    % distributing the Boltzmann weights evenly over all vertices.
    % 
    %     Parameters
    %     ----------
    %     beta : float
    %         Inverse temperature.
    %     J : float
    %         Coupling strength.
    % 
    %     Returns
    %     -------
    %     O : np.array (2, 2, 2, 2)
    %         MPO tensor,
    %         ordered left-top-right-bottom.
    
    vertex = zeros(repmat(2, 1, 4));
    for i = 1:2
        sbs = num2cell(repmat(i, 1, 4));
        vertex(sbs{:}) = 1;
    end
    c = sqrt(cosh(beta*J)); s = sqrt(sinh(beta*J));
    Qsqrt = 1 / sqrt(2) * [c+s, c-s; c-s, c+s];
    O = ncon({Qsqrt, Qsqrt, Qsqrt, Qsqrt, vertex}, {[-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]});
end


function M = isingM(beta, J)
    % Gives the magnetizatopn MPO tensor for the 2d classical Ising model at a
    % given temperature and coupling.
    % 
    %     Parameters
    %     ----------
    %     beta : float
    %         Inverse temperature.
    %     J : float
    %         Coupling strength.
    % 
    %     Returns
    %     -------
    %     M : np.array (2, 2, 2, 2)
    %         Magnetization MPO tensor,
    %         ordered left-top-right-bottom.

    vertex = zeros(repmat(2, 1, 4));
    for i = 1:2
        sbs = num2cell(repmat(i, 1, 4));
        vertex(sbs{:}) = 1;
    end
    Z = [1, 0; 0, -1];
    c = sqrt(cosh(beta*J));	s = sqrt(sinh(beta*J));
    Qsqrt = 1 / sqrt(2) * [c+s, c-s; c-s, c+s];
    vertexZ = ncon({Z, vertex}, {[-1,1], [1,-2,-3,-4]});
    M = ncon({Qsqrt, Qsqrt, Qsqrt, Qsqrt, vertexZ}, {[-1,1], [-2,2], [-3,3], [-4,4], [1,2,3,4]});
end


function M = isingMagnetization(beta, J, Ac, Fl, Fr)
    % Computes the expectation value of the magnetization in the Ising model
    % for a given temperature and coupling
    % 
    %     Parameters
    %     ----------
    %     beta : float
    %         Inverse temperature.
    %     J : float
    %         Coupling strength.
    %     Ac : np.array(D, d, D)
    %         MPS tensor of the MPS fixed point,
    %         with 3 legs ordered left-bottom-right,
    %         center gauge.
    %     Fl : np.array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    %     Fr : np.array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.
    % 
    %     Returns
    %     -------
    %     M : float
    %         Expectation value of the magnetization at the given temperature
    %         and coupling.

    M = ncon({Fl, Ac, isingM(beta, J), conj(Ac), Fr}, {[1, 2, 4], [4, 5, 6], [2, 5, 7, 3], [1, 3, 8], [6, 7, 8]});
end


function Z = isingZ(beta, J, Ac, Fl, Fr)
    % Computes the Ising model partition function for a given temperature and
    % coupling
    % 
    %     Parameters
    %     ----------
    %     beta : float
    %         Inverse temperature.
    %     J : float
    %         Coupling strength.
    %     Ac : np.array(D, d, D)
    %         MPS tensor of the MPS fixed point,
    %         with 3 legs ordered left-bottom-right,
    %         center gauge.
    %     Fl : np.array(D, d, D)
    %         left fixed point,
    %         ordered bottom-middle-top.
    %     Fr : np.array(D, d, D)
    %         right fixed point,
    %         ordered top-middle-bottom.
    % 
    %     Returns
    %     -------
    %     Z : float
    %         Value of the partition function at the given temperature and
    %         coupling.
    
    Z = ncon({Fl, Ac, isingO(beta, J), conj(Ac), Fr}, {[1, 2, 4], [4, 5, 6], [2, 5, 7, 3], [1, 3, 8], [6, 7, 8]});
end


function [magnetization, free, energy] = isingExact(beta, J)
    % Exact Onsager solution for the 2d classical Ising Model
    % 
    %     Parameters
    %     ----------
    %     beta : float
    %         Inverse temperature.
    %     J : float
    %         Coupling strength.
    % 
    %     Returns
    %     -------
    %     magnetization : float
    %         Magnetization at given temperature and coupling.
    %     free : float
    %         Free energy at given temperature and coupling.
    %     energy : float
    %         Energy at given temperature and coupling.

    theta = 0:1e-6:pi/2;
    x = 2*sinh(2*J*beta)/cosh(2*J*beta)^2;
    if 1-(sinh(2*J*beta))^(-4)>0
        magnetization = (1-(sinh(2*J*beta))^(-4))^(1/8);
    else
        magnetization = 0;
    end
    free = -1/beta*(log(2*cosh(2*J*beta))+1/pi*trapz(theta,log(1/2*(1+sqrt(1-x^2*sin(theta).^2)))));
    K = trapz(theta,1./sqrt(1-x^2*sin(theta).^2));
    energy = -J*cosh(2*J*beta)/sinh(2*J*beta)*(1+2/pi*(2*tanh(2*J*beta)^2-1)*K);
end


%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUXILIARY FUNCTIONS, from chapters 1 and 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Al, Ac, Ar, C] = minAcC(AcTilde, CTilde)
        
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

function A =  createMPS(D, d)
    A = rand(D, d, D) + 1i * rand(D, d, D);
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
