% initialize the population
% Create a population of N individuals
function f = ini_population(N)

%data = xlsread('F:\NAGS2\quanbu2.xlsx');
%[r,c] = size(data);
% V represents the number of features, which is also the dimension of each chromosome
% L means the number of 1's and also the number of features to select
% M represents the dimension of the evaluation function
% c = 229;  v = c-1;
global V;
L = 10;
M = 2;

for i = 1:N
    f(i,:) = [ones(1,L), zeros(1,V-L)];
    f(i,:) = f(i,(randperm(V)));
    %f(i,:) = randperm(f(i,:))
    
    temp(i,:) = evaluate_chrome(f(i,:));
end
f(:,(V+1):(V+M)) = temp;
end

