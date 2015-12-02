function [ indices ] = knn( origin_feat, centroids )
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
    [distances,indices]=min(Distance,[],2); %z distance, g_cur belonging cluster
end

