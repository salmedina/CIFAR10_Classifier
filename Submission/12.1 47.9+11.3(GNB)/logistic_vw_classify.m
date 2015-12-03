function [Y]=logistic_vw_classify(Model,X)
    %MULTINOMIAL LOGISTIC REGRESSION with Bag of Visual Words
    %Returns multi_theta and centroids (50 VW's)
    load('Model1.mat');
    C50=Model1{1};
    multi_theta=Model1{2};
    
    X_descriptors=double(extract_all_dsift(X,8)); %64 vectors per image
    X_descriptors=X_descriptors';
    X_vws=knn(X_descriptors,C50);
    X_bow=transform_data(X_vws,64,50);
    
    %add Intercept
    [m,n]=size(X_bow);
    X_bow = [ones(m,1), X_bow];
    
    %Calculate outcomes
    [prob, Y] = max(sigmoid(X_bow*multi_theta'),[],2);
    Y = uint8(Y-1); %Trained for class labels [1...10]
end