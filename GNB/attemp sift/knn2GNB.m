function [ new_feat ] = knn2GNB( indices, span, num_bins )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    size_data = size(indices, 1)/span;
    
    % get the word vectore
    temp_vec = cell(N, 1);
    for i = 1:N
        temp = cur_cluster((16*i - 15):(16*i), 1);
        temp = temp';
        temp_vec{i} = temp;
    end
    temp_vec = cell2mat(temp_vec); 
    
    new_feat = zeros(N, K);
    for i = 1:N
        A = temp_vec(i, :);
        % count the occurance
        A2 = unique(A);
        count = histc(A, A2);
        
        % set the new feature vector
        for k = 1:size(A2, 2)
            new_feat(i, A2(k)) = count(k);
        end
    end
    
    % normaliaze the feature vector
    for i = 1:N
        new_feat(i,:) = new_feat(i,:) ./ sqrt(sum(new_feat(i,:).^2));
    end
    %new_feat = normr(new_feat);

end

