
% 0 1 Binary Crossover Variation
function f = chrome_operator(parent_chromosome);

[N,M] = size(parent_chromosome);
%  V represents the number of features, which is also the dimension of each chromosome (1598 == M) where M == V + K + 1
%  L means the number of 1's and also the number of features to select
%  M represents the dimension of the evaluation function
global V;
L = 10;
M = 2;
% p 用来表示 交叉变异后产生的子代数
p = 1;
was_crossover = 0;
was_mutation = 0;
for i = 1 : N
    
    if rand(1) < 0.8
        child_1 = [];
        child_2 = [];
        
        parent_1 = randi([1,N]);
        parent_2 = randi([1,N]);
        
        while isequal(parent_chromosome(parent_1,:),parent_chromosome(parent_2,:))
            parent_2 = randi([1,N]);
        end
        parent_1 = parent_chromosome(parent_1,1:V);
        parent_2 = parent_chromosome(parent_2,1:V);
        
        for j = 1:V
            if rand(1) > 0.5
                temp = parent_1(j);
                parent_1(j) = parent_2(j);
                parent_2(j) = temp;
            end
            child_1(j) = parent_1(j);
            child_2(j) = parent_2(j);
        end
        child_1(:,V + 1: V+M) = evaluate_chrome(child_1);
        child_2(:,V + 1: V+M) = evaluate_chrome(child_2);
        was_crossover = 1;
        was_mutation = 0;
        %parent_1_index = (find(parent_1 == 1));
        %parent_2_index = (find(parent_2 == 1));
    else
        parent_3 = randi([1,N]);
        child_3 = parent_chromosome(parent_3,1:V);
        for j = 1:V
            if rand(1)>0.8
                child_3(j) = 1 - child_3(j);
            end
        end
        child_3(:,V + 1: V + M) = evaluate_chrome(child_3);
        was_mutation = 1;
        was_crossover = 0;
    end
    
    if was_crossover
        child(p,:) = child_1;
        child(p+1,:) = child_2;
        was_cossover = 0;
        p = p + 2;
    elseif was_mutation
        child(p,:) = child_3;
        was_mutation = 0;
        p = p + 1;
    end
end
f = child;
end
        
        
        
