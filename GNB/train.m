function [p, mean, var] = train(data, labels)
% X contains the training data, Y contains the training labels corresponding to X, 
% Model is a struct with all the parameters that define the model 
% (you may design the struct to contain any parameters that are useful to your model). 

% The dimensions of X are Ntrain × D and dimensions of Y are Ntrain × 1. 
% where Ntrain is the number of data points in the training data 
% and D is the dimension of each point (number of features).
     
% GNB implementation
    features = data2features(data); % geth the feature matrix
    p = prior(labels); % get the prior matrix
    [mean, var] = likelihood(features, labels); % get the mean and var
end

