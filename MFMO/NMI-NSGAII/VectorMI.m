
function mi = VectorMI(u1,u2,wind_size)

% Calculate the mutual information of two vector

% the size of u1 and u2 must be equal

% wind_size:size of the window

if nargin<2

disp('Error: please input at least two parameters:u1, u2');

return

end

if nargin==2

wind_size=floor(power(length(u1),1/3)+0.5);

end

x = [u1, u2];

n = wind_size;

[xrow, xcol] = size(x);

bin = zeros(xrow,xcol);

pmf = zeros(n, 2);

for i = 1:2

minx = min(x(:,i));

maxx = max(x(:,i));

binwidth = (maxx - minx) / n;

edges = minx + binwidth*(0:n);

histcEdges = [-Inf edges(2:end-1) Inf];

[occur,bin(:,i)] = histc(x(:,i),histcEdges,1);

pmf(:,i) = occur(1:n)./xrow;

end

% Calculate the joint probability density of u1 and u2

jointOccur = accumarray(bin,1,[n,n]);

%(xi?yi) The number of two data falling into the n * n aliquot is the joint probability density

jointPmf = jointOccur./xrow;

Hx = -(pmf(:,1))'*log2(pmf(:,1)+eps);

Hy = -(pmf(:,2))'*log2(pmf(:,2)+eps);

Hxy = -(jointPmf(:))'*log2(jointPmf(:)+eps);

MI = Hx+Hy-Hxy;

% mi = MI/sqrt(Hx*Hy);
mi = (2*MI)/(Hx+Hy);
