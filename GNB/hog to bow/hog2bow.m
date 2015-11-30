function [ new_feat, centroids ] = hog2bow( origin_feat, K )
% get the original HOG features, and turn that into BOW features

    N = size(origin_feat, 1);
    % process the features first
    temp_feat = cell(16, 1);  
    for i = 1:16
        temp_feat{i} = origin_feat(:, (31*i - 30):(31*i));
    end
    temp_feat = cell2mat(temp_feat);
    
    disp('After turning origin features to temp features');
    % then get the centroids and the concatenate word vecter
    [cur_cluster, centroids] = k_means(temp_feat, K);
    disp('After K-Means');
    % get the word vectore
    temp_vec = cell(N, 1);
    for i = 1:N
        temp = cur_cluster((16*i - 15):(16*i), 1);
        temp = temp';
        temp_vec{i} = temp;
    end
    temp_vec = cell2mat(temp_vec); 
    disp('After getting temp vectores N*16');
    new_feat = zeros(N, K);
    for i = 1:size(origin_feat)
        A = temp_vec(i, :);
        % count the occurance
        A2 = unique(A);
        count = histc(A, A2);
        
        % set the new feature vector
        for k = 1:size(A2, 2)
            new_feat(i, A2(k)) = count(k);
        end
    end
    disp('After getting new features N*K');
    % normaliaze the feature vector
    new_feat = normr(new_feat);
end

