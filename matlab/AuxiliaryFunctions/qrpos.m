function [Q,R]=qrpos(A,a)

if nargin<2
    [Q,R]=qr(A,0);
else
    [Q,R]=qr(A,a);
end

if size(Q,1)==1
    Q=Q*sign(R(1));
    R=R*sign(R(1));
else
    D=diag(R);
    D(abs(D)<1e-10)=1;
    D=sign(D);
    Q=Q.*D';
    R=D.*R;
end

end