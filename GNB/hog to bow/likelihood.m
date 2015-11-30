function [Mean, Var] = likelihood (features, labels)

% M is an m × c matrix where Mi,j is the conditional mean of feature i given class j. 
% V is an m × c matrix where Vi,j is the conditional variance of feature i given class j.
F = size(features, 2);
Mean = zeros(F, 10); % the number of classes is fixed
Var = zeros(F, 10); % the number of classes is fixed

%iterate over the features
for i = 1:F
    feat = features(:, i); % get the i-th feature vector
    merge = [double(labels), feat]; % merge to ease the calculation
    %iterate over the classes
    for j = 1:10
        c_j_vec = merge(:, 1) == (j-1); % get the location vector of all items labeled j
        c_j = merge(c_j_vec, :); % get the matrix of all items labeled j
        mean = sum(c_j(:,2)) / (size(c_j, 1)); % get the mean
        Mean(i, j) = mean;
        
        c_j_feat = c_j(:, 2); % get the feature vector without label
        var = sum((c_j_feat - mean).^2) / (size(c_j, 1)); % get the variance
        Var(i, j) = var;
    end
end

end