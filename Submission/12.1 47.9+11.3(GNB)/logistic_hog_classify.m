function [Y]=logistic_hog_classify(Model,X)
% Uses the pretrained logistic classifier based on HOG features with
% no whitening
    %Returns theta for HoG
    load('Model1.mat');
    theta=Model1{1};
    
    X_descriptors=extract_all_hog(X,8);
    
    %add Intercept
    [m,n]=size(X_descriptors);
    X_descriptors = [ones(m,1), X_descriptors];
    
    %Calculate outcomes
    [prob, Y] = max(sigmoid(X_descriptors*theta'),[],2);
    Y = uint8(Y-1); %Trained for class labels [1...10]

end