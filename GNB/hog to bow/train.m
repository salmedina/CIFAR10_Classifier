function [Model] = train(data, labels, k)
% X contains the training data, Y contains the training labels corresponding to X, 
% Model is a struct with all the parameters that define the model 
% (you may design the struct to contain any parameters that are useful to your model). 

% The dimensions of X are Ntrain × D and dimensions of Y are Ntrain × 1. 
% where Ntrain is the number of data points, origin_feat, new_fea in the training data 
% and D is the dimension of each point (number of features).
     
% GNB implementation
    origin_feat = data2features(data); % geth the feature matrix
    %disp('After data2features');
    [ new_feat, centroids ] = hog2bow(origin_feat, k);
    %disp('After hog2bow');

    p = prior(labels); % get the prior matrix
    %disp('After prior');
    %[mean, var] = likelihood(features, labels); % get the mean and var
    [mean, var] = likelihood(new_feat, labels);
    %disp('Finish');
    
    Model = cell(4,1);
    Model{1} = p;
    Model{2} = mean;
    Model{3} = var;
    Model{4} = centroids;
end

