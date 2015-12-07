function [x_zca, x_mean, x_eigenvect] = zca_whiten(x, epsilon)
%ZCA whitening, first it normalizes with respect to the mean
%then sets the variance to 1 by obtaining the eigen-vectors/values
%through SVD and finally smooths the rotation over the axis
%through a low-pass filtering epsilon
    x_mean = mean(x, 1);  %calc mean of each dimension/feature
    x = double(x) - repmat(x_mean,size(x,1), 1); %shift all features to mean0
    x_sigma = x * x' / size(x,2); %calculate the covar matrix
    [x_eigenvect, x_eigenvals, ~] = svd(x_sigma);
    %rotate the data with low-filter value
    x_zca = x_eigenvect * diag(1./sqrt(diag(x_eigenval) + epsilon)) * x_eigenvect' * x;
end