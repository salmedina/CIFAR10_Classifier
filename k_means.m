function [cur_cluster,centroids]=k_means(X,k)
% K_MEANS    k-means clustring

[n,m]=size(X);  %n samples, m features

% Input can be:
% 1. a list of initialized centroids  OR
% 2. a scalar indicating the number of centroids
if ~isscalar(k)
    centroids=k;
    k=size(centroids,1);
else
    centroids=X(ceil(rand(k,1)*n),:); %Initialize at random the centroids
end

% auxiliary vars
prev_cluster=ones(n,1);  %previous group to which elemes belong
cur_cluster=zeros(n,1);  %current group to which elemes belong
Distance=zeros(n,k);      %Distance matrix from each elem to each centroid

% Main loop converge if previous partition is the same as current
while any(prev_cluster~=cur_cluster)  %TODO: possible error if elements keeps on switching groups
    prev_cluster=cur_cluster;
    % Calculate the distance from each point to each centroid
    for t=1:k
        % Calculate the euclidean distance in parallel 
        %for all elems vs all centroids
        d=zeros(n,1);
        for s=1:m
            d=d+(X(:,s)-centroids(t,s)).^2;
        end
        Distance(:,t)=d;
    end
    % Get belonging cluster
    [z,cur_cluster]=min(Distance,[],2); %z distance, g_cur belonging cluster
    
    % Update centroids
    for t=1:k
        centroids(t,:)=mean(X(cur_cluster==t,:)); %filter X through cur cluster
    end
end
