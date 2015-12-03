function [ indices ] = knn( origin_feat, centroids )
% turn the test set into bow features

    M = size(origin_feat,1); % number of samples
    K = size(centroids, 1); % number of centroids
    Distance = zeros(M, K);
    for t = 1:K
         % Calculate the euclidean distance in parallel 
        %for all elems vs all centroids
        d = zeros(M,1);
        for s=1:size(origin_feat, 2)
            d=d+(origin_feat(:,s)-centroids(t,s)).^2;
        end
        Distance(:,t)=d;
    end
    
    % Get belonging cluster
    [distances,indices]=min(Distance,[],2); %z distance, g_cur belonging cluster
end

