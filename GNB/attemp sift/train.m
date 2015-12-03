function [Model] = train(data, labels)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    descriptors = extract_all_dsift(data, 8);
    descriptors = descriptors';
    descriptors = double(descriptors);
    [cur_cluster, centroids] = kmeans(descriptors, 10);
    %[cur_cluster, centroids] = k_means(descriptors, 10);
    bow_data = transform_data(cur_cluster, 64, 10);

    p = prior(labels); % get the prior matrix
    [mean, var] = likelihood(bow_data, labels);
    %disp('Finish');
    
    Model = cell(4,1);
    Model{1} = p;
    Model{2} = mean;
    Model{3} = var;
    Model{4} = centroids;
end

