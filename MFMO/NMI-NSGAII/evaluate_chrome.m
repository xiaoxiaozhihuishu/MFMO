
% Evaluation function to be optimized
% chrome is chromosome
% chrome is a feature of 0 1 0 1 representing a binary vector, representing a subset of selected features
function f = evaluate_chrome(chrome)

% data = xlsread('gradient_.xlsx');
global data;
[r,c] = size(data);
label = data(:,c);
value = data(:,1:(c-1));
row = size(chrome,1);
non_zero_num = sum(chrome);

for i = 1:row
    temp = find(chrome==1);
    temp = value(:,temp);
    
    Max_Relevance = 0;
    Min_Redundancy = 0;
    
    len = size(temp,2);
    for i = 1:len
        Max_Relevance = Max_Relevance + (1 - VectorMI( temp(:,i) , label ));
        for j = 1:len
            Min_Redundancy = Min_Redundancy + VectorMI( temp(:,i) , temp(:,j) );
            %disp(VectorMI( temp(:,i) , temp(:,j)))
        end
    end
    Max_Relevance = Max_Relevance / len;
    Min_Redundancy = Min_Redundancy / power(len,2);
    
    f(1) = Max_Relevance;
    f(2) = Min_Redundancy;
end

end

