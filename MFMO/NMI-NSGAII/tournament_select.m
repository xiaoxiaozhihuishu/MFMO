function f = tournament_select(chromosome,pool_size,tour_size)

[pop,variables] = size(chromosome);
rank = variables - 1;
distance = variables;

for i = 1 : pool_size
    % Randomly find tour_size objects
    for j = 1 : tour_size
        candidate(j) = round(pop*rand(1));
        if candidate(j) == 0
            candidate(j) = 1;
        end
        if j > 1
            while ~isempty(find(candidate(1 : j - 1) == candidate(j)))
                candidate(j) = round(pop*rand(1));
                if candidate(j) == 0
                    candidate(j) = 1;
                end
            end
        end
    end
    
    % store rank and distance
    for j = 1:tour_size
        c_obj_rank(j) = chromosome(candidate(j),rank);
        c_obj_distance(j) = chromosome(candidate(j),distance);
    end
    
    % find the best chromosome (candidate)
    min_candidate = ...
        find(c_obj_rank == min(c_obj_rank));
    if length(min_candidate) ~= 1
        max_candidate = ...
            find(c_obj_distance(min_candidate) == max(c_obj_distance(min_candidate)));
        if length(max_candidate) ~= 1
             max_candidate = max_candidate(1);
        end
        f(i,:) = chromosome(candidate(min_candidate(max_candidate)),:);
    else
        f(i,:) = chromosome(candidate(min_candidate(1)),:);
    end
end
end
        
            
