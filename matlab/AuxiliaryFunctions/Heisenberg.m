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


