function [ new_feat ] = knn( origin_feat, centroids )
% turn the test set into bow features

    N = size(origin_feat, 1); 
    K = size(centroids, 1); % number of centroids
    % process the features first
    temp_feat = cell(16, 1);  
    for i = 1:16
        temp_feat{i} = origin_feat(:, (31*i - 30):(31*i));
    end
    temp_feat = cell2mat(temp_feat);
    
    Distance = zeros(size(temp_feat, 1), K);
    for t = 1:K
         % Calculate the euclidean distance in parallel 
        %for all elems vs all centroids
        d = zeros(size(temp_feat, 1),1);
        for s=1:size(temp_feat, 2)
            d=d+(temp_feat(:,s)-centroids(t,s)).^2;
        end
        Distance(:,t)=d;
    end
    
    % Get belonging cluster
    [z,cur_cluster]=min(Distance,[],2); %z distance, g_cur belonging cluster
   
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
        new_feat(i,:) = new_feat(i,:) ./ sum(new_feat(i,:));
    end
    %new_feat = normr(new_feat);
end

