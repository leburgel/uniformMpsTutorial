function y=ArrayIsEqual(x1,x2,tol)

if nargin==2 || isempty(tol)
    tol=eps;
end

y=ArrayNorm(x1-x2)<tol;

end