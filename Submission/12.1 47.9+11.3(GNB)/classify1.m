function [Y]=classify(Model,X)
    %Returns multi_theta and centroids (50 VW's)
    load('Model1.mat');
    simgoid = @(z) 1.0/(1.0+exp(-z)); 
    
    X_descriptors=extract_all_dsift(X,8); %64 vectors per image
    X_vws=knn(X,C50);
    X_bow=transform_data(X_vws,64,50);
    
    %add Intercept
    [m,n]=size(X_bow);
    X_bow = [ones(m,1), X_bow];
    
    %Calculate outcomes
    [prob, Y] = max(sigmoid(X*multi_theta'),[],2);
    Y = Y-1; %Trained for class labels [1...10]
end