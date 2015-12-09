function [X_zca, zca] = zca_whiten(X, epsilon)
%ZCA whitening, first it normalizes with respect to the mean
%then sets the variance to 1 by obtaining the eigen-vectors/values
%through SVD and finally smooths the rotation over the axis
%through a low-pass filtering epsilon

    X_mean = mean(X, 1);  %calc mean of each dimension/feature
    X = double(X) - repmat(X_mean,size(X,1), 1); %shift all features to mean0
    
    X_sigma = X * X' / size(X,2); %calculate the covar matrix
    
    [X_eigenvect, X_eigenval, ~] = svd(X_sigma); %calculate eigen*
    
    %rotate the data with low-filter value
    X_zca = X_eigenvect * diag(1./sqrt(diag(X_eigenval) + epsilon)) * X_eigenvect' * X;
    zca = struct('mean',X_mean, 'eigenval',X_eigenval,'eigenvect',X_eigenvect);
end