% input the filter feature matrix which is the sample * feature
% Output feature selection matrix, each row of the matrix is a 0-1 binary vector
% each row of the matrix represents a feature group
% in the feature group, 1 represents the feature is selected
% 0 represents the feature is selected
profile on
% M represents the dimension of the evaluation function
M = 2;
% V is the number of the features
pop = 100;
global data;
global V;
data = xlsread('filter_features.xlsx');
V = size(data,2)-1;
q1 = real(ini_population(pop));
% a = zeros(1,V);
% a(1:10) = 1;
% temp = evaluate_chrome(a);
% a(V+1:V+M)=temp;
% q1(1,:) = a;
q2 = n_d_sort(q1);
temp_num = zeros(1,V);
for i = 1:50
    
    po = round(pop/2);
    tour = 2;
    
    q3 = tournament_select(q2,po,tour);
    
    q4 = real(chrome_operator(q3));
    
    [main_pop,temp] = size(q2);
    [offspring_pop,temp] = size(q4);
    intermediate_chromosome(1:main_pop,:) = q2;
    
    intermediate_chromosome(main_pop + 1 : main_pop + offspring_pop,1 : M+V) = q4;
    intermediate_chromosome = n_d_sort(intermediate_chromosome);
    
    q2 =real(new_chrome(intermediate_chromosome,pop));
    display(i);
    
    %if(sum(q2(:,M+V+1))==100)
    %    temp_num = temp_num + 1;
    %end
    
    %if(temp_num ==25)
    %    break
    %end
    temp_a = q2(1:100,1:V);
    temp_b = sum(temp_a);
    temp_c = 0;
    
    for j = 1:V
        if temp_b(j) ~= 0
            temp_c = temp_c + 1;
        end
    end
    temp_num(i) = temp_c;
    
end
profile viewer
p = profile('info');
% temp = find(aa == 1);
NSAG_sel = q2(1:100,1:V);
xlswrite("NSGA2_sel.xlsx",NSAG_sel);
