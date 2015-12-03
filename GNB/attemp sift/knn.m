function [ indices ] = knn( descriptors, centroids )
% turn the test set into bow features

    N = size(descriptors, 1); 
    K = size(centroids, 1); % number of centroids
    descriptors = double(descriptors);
    centroids = double(centroids);
    
    Distance = zeros(N, K);
    for t = 1:K
        % Calculate the euclidean distance in parallel 
        %for all elems vs all centroids
        d = zeros(N,1);
        for s=1:size(descriptors, 2)
            d=d+(descriptors(:,s)-centroids(t,s)).^2;
        end
        Distance(:,t)=d;
    end
    
    % Get belonging cluster
    [z,indices]=min(Distance,[],2); %z distance, g_cur belonging cluster
end

