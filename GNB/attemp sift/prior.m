function [p] = prior(labels)

%max_ele = max(max(labels)); % get the value of c, here the max should be 9
p = zeros(10, 1); % init a matrix for prior

%for i = 0:max_ele
for i = 0:9
    logical = (labels == i); % get a vector marking all ele equal to i
    p(i+1) = sum(logical) / length(labels); % get the possibility of class i
end

end
