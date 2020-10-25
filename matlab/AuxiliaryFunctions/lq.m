function [q,l]=lq(x,choice)

if nargin==2
    [q,l]=qrpos(x.',choice);
    q=q.'; l=l.';
elseif nargin==1
    [q,l]=qrpos(x.');
    q=q.'; l=l.';
end

end